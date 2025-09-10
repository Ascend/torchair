from collections import deque, OrderedDict
from typing import List, Optional, Callable, Any, Deque, Dict, Set, Tuple, Union
from dataclasses import dataclass, field
import functools
import gc
import os
import pickle
import sys
import sympy
import warnings

import torch
from torch import fx
from torch import nn
from torch.fx import Node, Proxy
from torch.profiler import record_function

from torchair.core.utils import logger
from torchair.scope._scope_attr import guard_with_user_stream_scope
from torchair._acl_concrete_graph.utils import reconstruct_args_kwargs
from torchair._acl_concrete_graph.utils import (debug_mem_state, WeakRef, GraphMeta, get_tensor_metadata,
                                                reconstruct_from_tensor_metadata, reconstruct_args_kwargs)
from torchair._acl_concrete_graph.static_kernel import compile_static_kernel


@dataclass
class StaticWorkspaceReplaceFunc:
    """
    Data class defining replacement functions for static workspace operations.

    Attributes:
        get_workspace (Callable): Function to retrieve workspace size.
        out_operator (Callable): Replacement operator for the original operation.
        workspace_keys (List[str]): Keys for workspace parameters.
        output_keys (List[str]): Keys for output parameters.
        updated_param_keys (List[str]): Parameters requiring updates.
    """
    get_workspace: Callable
    out_operator: Callable
    workspace_keys: List[str]
    output_keys: List[str]
    updated_param_keys: List[str]


@dataclass
class UpdatedNodeInfo:
    """
    Information about updated nodes during graph capture.

    Attributes:
        node_name (str): Name of the updated node.
        updated_func (Callable): Function performing the update.
        updated_param_name (List[str]): Names of parameters being updated.
        args (Any): Arguments passed to the update function.
        kwargs (Any): Keyword arguments passed to the update function.
        handle (Any): Handle to the graph task group.
        event (Any): Event signaling completion of the update.
    """
    node_name: str
    updated_func: Callable
    updated_param_name: List[str]
    args: Any
    kwargs: Any
    handle: Any
    event: Any


_REPLACE_FUNC_MAP = {
    torch.ops.npu.npu_fused_infer_attention_score.default: StaticWorkspaceReplaceFunc(
        get_workspace=torch.ops.npu._npu_fused_infer_attention_score_get_max_workspace.default,
        out_operator=torch.ops.npu.npu_fused_infer_attention_score.out,
        workspace_keys=["workspace"],
        output_keys=["attention_out", "softmax_lse"],
        updated_param_keys=["actual_seq_lengths", "actual_seq_lengths_kv", "actual_shared_prefix_len"],
    ),
}
if hasattr(torch.ops.npu, "npu_fused_infer_attention_score_v2"):
    _REPLACE_FUNC_MAP.update({torch.ops.npu.npu_fused_infer_attention_score_v2.default: StaticWorkspaceReplaceFunc(
        get_workspace=torch.ops.npu._npu_fused_infer_attention_score_v2_get_max_workspace.default,
        out_operator=torch.ops.npu.npu_fused_infer_attention_score_v2.out,
        workspace_keys=["workspace"],
        output_keys=["attention_out", "softmax_lse"],
        updated_param_keys=["actual_seq_qlen", "actual_seq_kvlen"],
    )})


def is_constant(arg):
    return isinstance(arg, (int, float, bool))


def is_sym(arg):
    return isinstance(arg, (torch.SymInt, torch.SymFloat, torch.SymBool))


def have_sym_in_list(arg):
    if not isinstance(arg, (list, tuple)) or len(arg) == 0:
        return False

    for arg_i in arg:
        if is_sym(arg_i):
            return True

    return False


def have_sym_in_meta(node_meta):
    if isinstance(node_meta, torch.Tensor):
        return have_sym_in_list(list(node_meta.size()))
    elif isinstance(node_meta, (list, tuple)):
        return have_sym_in_list(node_meta)
    else:
        if is_sym(node_meta):
            return True
    return False


def get_node_all_placeholder_inputs(node, excluded_kwargs=None):
    if not isinstance(node, fx.Node):
        return set()

    excluded_kwargs = [] if excluded_kwargs is None else excluded_kwargs
    placeholders = set()
    processing_input_queue: Deque[Any] = deque()
    processing_input_queue.extend(node.args)
    for kwarg_name, kwarg_value in node.kwargs.items():
        if kwarg_name not in excluded_kwargs:
            processing_input_queue.append(kwarg_value)

    while processing_input_queue:
        cur_input = processing_input_queue.popleft()
        if isinstance(cur_input, Node):
            if cur_input.op == "placeholder":
                placeholders.add(cur_input)
        elif isinstance(cur_input, (list, tuple)):
            processing_input_queue.extend(cur_input)
        elif isinstance(cur_input, dict):
            processing_input_queue.extend(cur_input.valus())
        elif isinstance(cur_input, (int, float, bool, str, None)):
            # constant value, must not be a placeholder
            pass
        else:
            raise RuntimeError(f"Current node name[{node.name}] "
                               f"input type {type(cur_input)} is unsupported, with value:{cur_input}.")

    return placeholders


def gen_unupdate_input_func(unupdated_input_index: List):
    if len(unupdated_input_index) == 0:
        def empty_input_key(*args: Any, **kwargs: Any):
            return "no_updated_input"

        return empty_input_key

    def gen_unupdated_input_key(*args: Any, **kwargs: Any):
        input_shape_list = []
        for idx in unupdated_input_index:
            if isinstance(args[idx], torch.Tensor):
                input_shape_list.append(str(list(args[idx].shape)))
            else:
                input_shape_list.append(str(args[idx]))
        return ",".join(input_shape_list)

    return gen_unupdated_input_key


def get_unupdated_sym_input_index(graph_module: torch.fx.GraphModule):
    updated_op_params = {}
    for func_iter in _REPLACE_FUNC_MAP.values():
        if len(func_iter.workspace_keys) > 0:
            updated_op_params[func_iter.get_workspace] = func_iter.updated_param_keys
        updated_op_params[func_iter.out_operator] = func_iter.updated_param_keys
    logger.debug("In graph[%s], all updated inputs user nodes and params: %s.", id(graph_module), updated_op_params)

    unupdated_sym_input_index = []
    data_idx = -1
    for node in graph_module.graph.nodes:
        if node.op != "placeholder":
            continue
        data_idx = data_idx + 1
        if not hasattr(node, "meta"):
            # int placeholder does not have 'meta' attr or symbol, skip this case
            logger.debug('Find no meta attr placeholder node, placeholder index=%s, value=%s, type=%s',
                         data_idx, node, type(node).__name__)
            continue
        node_meta = node.meta['val']
        if not have_sym_in_meta(node_meta):
            continue
        logger.debug("In graph[%s], the %s th meta input[%s] have sym, with all users[%s].",
                     id(graph_module), data_idx, node_meta, node.users)
        if len(node.users) == 0:
            continue

        have_unupdated_user = False
        for user_node in node.users:
            if user_node.target not in updated_op_params.keys():
                have_unupdated_user = True
                break
            unupdated_inputs = get_node_all_placeholder_inputs(user_node,
                                                               excluded_kwargs=updated_op_params[user_node.target])
            logger.debug("In graph[%s], the %s th meta input[%s] have unupdated user[user_name: %s, user_inputs:%s].",
                         id(graph_module), data_idx, node_meta, user_node.name, unupdated_inputs)
            if node in unupdated_inputs:
                have_unupdated_user = True
                break
        if have_unupdated_user:
            unupdated_sym_input_index.append(data_idx)
    logger.debug("In graph[%s], all unupdated sym input index is %s.",
                 id(graph_module), unupdated_sym_input_index)

    return unupdated_sym_input_index


def get_unupdated_input_fn(unupdated_sym_input_index):
    return gen_unupdate_input_func(unupdated_sym_input_index)


def get_update_ruler(node, updated_param_keys, placeholder_nodes):
    update_rulers = {}
    for kwarg_name, kwarg_value in node.kwargs.items():
        if kwarg_name not in updated_param_keys:
            continue
        if not isinstance(kwarg_value, (list, tuple)):
            raise RuntimeError(f"For updated param type only list is supported, but get [{kwarg_value}].")

        update_ruler = []
        have_sym_in_param = False
        for kwarg_i in kwarg_value:
            if kwarg_i in placeholder_nodes:
                input_index = placeholder_nodes.index(kwarg_i)
                update_ruler.append(("index", input_index))
                have_sym_in_param = True
                continue

            if not is_constant(kwarg_i):
                raise RuntimeError(f"For updated param value only sym and constant is supported, "
                                   f"but get [{kwarg_i}].")
            update_ruler.append(("fixed", kwarg_i))
        logger.debug("Current node name[%s] %s to update kwargs, kwargs name[%s] gen value ruler[%s].",
                     node.name, "need" if have_sym_in_param else "no need", kwarg_name, update_ruler)

        if have_sym_in_param:
            update_rulers[kwarg_name] = update_ruler
    return update_rulers


def gen_updated_input_func(ops_update_rulers: Dict):
    if len(ops_update_rulers) == 0:
        def func(*args: Any, **kwargs: Any):
            return {}

        return func

    def func(*args: Any, **kwargs: Any):
        all_update_dict = {}
        for op_name, rulers in ops_update_rulers.items():
            op_update_dict = {}
            for param_name, ruler in rulers.items():
                param_list = [args[iter[1]] if iter[0] == "index" else iter[1] for iter in ruler]
                op_update_dict[param_name] = param_list
            all_update_dict[op_name] = op_update_dict
        return all_update_dict

    return func


def get_updated_ops_rulers_param(graph_module: torch.fx.GraphModule, meta_inputs: List):
    logger.debug("Start process inputs in graph[%s], with all meta inputs[%s].", id(graph_module), meta_inputs)

    placeholder_nodes = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
    if len(placeholder_nodes) != len(meta_inputs):
        raise RuntimeError(
            f'The lengths of the the placeholder nodes {len(placeholder_nodes)} and '
            f'the recorded meta inputs {len(meta_inputs)} do not match.')

    updated_dict = {}
    for func_iter in _REPLACE_FUNC_MAP.values():
        updated_dict[func_iter.out_operator] = func_iter.updated_param_keys
    logger.debug("In graph[%s], try to update node type and param: %s.", id(graph_module), updated_dict)

    ops_update_rulers = {}
    for node in graph_module.graph.nodes:
        if node.target not in updated_dict.keys():
            continue

        update_rulers = get_update_ruler(node, updated_dict[node.target], placeholder_nodes)
        if len(update_rulers) > 0:
            ops_update_rulers[node.name] = update_rulers
    logger.debug("All need to be updated param[%s] in graph[%s] .", ops_update_rulers, id(graph_module))

    need_updated_ops: Dict[str, List] = {}  # k: op_name, v: updated_param_name_list
    for op_name, update_rulers in ops_update_rulers.items():
        updated_params = [param_name for param_name, _ in update_rulers.items()]
        need_updated_ops[op_name] = updated_params
    logger.debug(" In graph[%s] all need to be updated node names[%s] param names[%s].",
                 id(graph_module), need_updated_ops.keys(), need_updated_ops.values())

    # no need to check all sym input must be in updated param indexes.
    return ops_update_rulers, need_updated_ops


def get_updated_ops_fn(ops_update_rulers):
    # no need to check all sym input must be in updated param indexes.
    return gen_updated_input_func(ops_update_rulers)


def check_all_sym_updated(ops_update_rulers: Dict, graph_module: torch.fx.GraphModule):
    all_updated_index = {}

    for op_name, rulers in ops_update_rulers.items():
        updated_index = set()
        for _, ruler in rulers.items():
            tmp_idx = {ruler_i[1] for ruler_i in ruler if ruler_i[0] == "index"}
            updated_index.update(tmp_idx)
        all_updated_index[op_name] = updated_index
    logger.debug("All updated input ops name and index is [%s].", all_updated_index)

    data_idx = -1
    for node in graph_module.graph.nodes:
        if node.op != "placeholder":
            continue
        data_idx = data_idx + 1
        if not is_sym(node.meta['val']):
            continue
        logger.debug("In graph[%s], the %s th meta input is sym[%s] with all users[%s].",
                     id(graph_module), data_idx, node.meta['val'], node.users)
        if len(node.users) == 0:
            continue

        is_updated = False
        for user_node in node.users:
            if user_node.name not in all_updated_index.keys():
                # TO DO: add check here
                continue

            if data_idx in all_updated_index[user_node.name]:
                is_updated = True
                break
        if not is_updated:
            raise RuntimeError(f"The {data_idx}th meta input is sym[{node.meta['val']}], "
                               f"and be used by nodes [{node.users}], "
                               f"which is not in updated index [{all_updated_index}].")


class UpdatedNodeCaptureInterp(fx.Interpreter):
    """
    Custom interpreter for capturing node updates during graph execution.
    """

    def __init__(self, graph_module: fx.GraphModule, need_updated_ops: Dict):
        super().__init__(graph_module)
        self._graph_module: fx.GraphModule = graph_module
        self._need_updated_ops: Dict[str, List] = need_updated_ops  # k: op_name, v: updated_param_name_list
        self._captured_node_info: List = []

    @property
    def captured_node_infos(self):
        return self._captured_node_info

    @guard_with_user_stream_scope
    def run_node(self, node):
        logger.debug("Try to capture node names[%s] type[%s] args[%s] kwargs[%s] in graph[%s] .",
                     node.name, node.target, node.args, node.kwargs, id(self._graph_module))

        if node.name not in self._need_updated_ops.keys():
            return super().run_node(node)

        # external event no need record before capture.
        external_event = torch.npu.ExternalEvent()
        capture_stream = torch.npu.current_stream()
        external_event.wait(capture_stream)
        external_event.reset(capture_stream)

        torch.npu.graph_task_group_begin(capture_stream)
        result = super().run_node(node)
        handle = torch.npu.graph_task_group_end(capture_stream)
        node_args, node_kwargs = self.fetch_args_kwargs_from_env(node)
        node_args, node_kwargs = reconstruct_args_kwargs(node_args, node_kwargs)

        self._captured_node_info.append(UpdatedNodeInfo(
            node_name=node.name,
            updated_func=node.target,
            updated_param_name=self._need_updated_ops[node.name],
            args=node_args,
            kwargs=node_kwargs,
            handle=handle,
            event=external_event)
        )
        logger.debug("Record the %s th updated node, node name[%s], node func[%s], node args length[%s], "
                     "node kwargs length[%s], update param name[%s], update task handle[%s], "
                     "update event[%s] in graph[%s].",
                     len(self._captured_node_info),
                     self._captured_node_info[-1].node_name,
                     self._captured_node_info[-1].updated_func,
                     len(self._captured_node_info[-1].args),
                     len(self._captured_node_info[-1].kwargs),
                     self._captured_node_info[-1].updated_param_name,
                     self._captured_node_info[-1].handle,
                     self._captured_node_info[-1].event,
                     id(self._graph_module)
                     )

        return result


class CapturedGraphUpdateAndReplay(nn.Module):
    """
    Module for replaying captured graphs with dynamic updates.
    """

    def __init__(self, replay_graph: Any, updated_input_func: Callable, updated_node_infos: List):
        super().__init__()
        self._replay_graph = replay_graph
        self._updated_input_func = updated_input_func
        self._updated_node_infos = updated_node_infos
        self._update_stream = torch.npu.Stream()
        self._current_stream = torch.npu.current_stream()

    def forward(self, *args: Any, **kwargs: Any):
        self._replay_graph.replay()

        updated_kwargs = self._updated_input_func(*args, **kwargs)
        logger.debug("In AclGraph running, all updated op_name and param: %s.", updated_kwargs)
        if len(updated_kwargs) == 0:
            return

        torch.npu.set_stream(self._update_stream)
        for node_info in self._updated_node_infos:
            torch.npu.graph_task_update_begin(self._update_stream, node_info.handle)
            node_kwargs = dict(node_info.kwargs)
            for key in node_info.updated_param_name:
                node_kwargs[key] = updated_kwargs[node_info.node_name][key]
            node_info.updated_func(*node_info.args, **node_kwargs)
            torch.npu.graph_task_update_end(self._update_stream)
            self._update_stream.record_event(node_info.event)
        torch.npu.set_stream(self._current_stream)

        logger.info("Replay AclGraph and update input params successfully.")
        return


def construct_and_add_workspace(node: fx.Node, graph_module: fx.GraphModule, kwargs_dict: dict) -> fx.Node:
    registered_workspace_len = len(_REPLACE_FUNC_MAP[node.target].workspace_keys)
    if registered_workspace_len == 0:
        logger.debug("Current node[%s] do not have workspace in registered info, skip construct and add workspace.",
                     node.target)
        return None

    workspace_node = graph_module.graph.call_function(_REPLACE_FUNC_MAP[node.target].get_workspace,
                                                      args=node.args, kwargs=node.kwargs)

    list_workspace_nodes = workspace_node if isinstance(workspace_node, (tuple, list)) else (workspace_node,)
    if len(list_workspace_nodes) != registered_workspace_len:
        raise RuntimeError(
            f'The lengths of the returned workspace nodes {len(list_workspace_nodes)} and '
            f'the register workspace keys {registered_workspace_len} do not match.')

    for i in range(registered_workspace_len):
        kwargs_dict[_REPLACE_FUNC_MAP[node.target].workspace_keys[i]] = list_workspace_nodes[i]

    return workspace_node


_global_sym_expr_2_node_map = {}


def construct_fx_node_shape(ori_shape: List, sym_inputs: dict):
    global _global_sym_expr_2_node_map
    empty_shape = []
    for dim in ori_shape:
        if is_constant(dim):
            empty_shape.append(dim)
        elif isinstance(dim.node.expr, sympy.Symbol):
            if dim.node.expr not in sym_inputs.keys():
                raise RuntimeError(f'Unexpected sym output shape {dim} which is not in graph '
                                   f'all sym inputs dict {sym_inputs}.')
            empty_shape.append(sym_inputs[dim.node.expr])
        else:
            if str(dim) in _global_sym_expr_2_node_map.keys():
                empty_shape.append(_global_sym_expr_2_node_map[str(dim)])
                continue
            sym_set = dim.node.expr.free_symbols
            proxy_nodes = {}
            for sym_i in sym_set:
                globals()[str(sym_i)] = Proxy(sym_inputs[sym_i])
            expr_proxy = eval(str(dim))
            empty_shape.append(expr_proxy.node)
            _global_sym_expr_2_node_map[str(dim)] = expr_proxy.node
            logger.debug("Record all created sym expr and fx node %s.", _global_sym_expr_2_node_map)

    logger.debug("Construct fx node shape %s from original meta shape %s .",
                 empty_shape, ori_shape)
    return empty_shape


def construct_and_add_output(node: fx.Node, graph_module: fx.GraphModule, kwargs_dict: dict,
                             sym_inputs: dict) -> fx.Node:
    registered_outputs_len = len(_REPLACE_FUNC_MAP[node.target].output_keys)
    if registered_outputs_len == 0:
        raise RuntimeError(f"Current node[{node.target}] do not have outputs in registered info, "
                           f"skip construct and add outputs.", )

    meta_outputs = node.meta['val']
    if not isinstance(meta_outputs, (tuple, list)):
        meta_outputs = (meta_outputs,)
    if len(meta_outputs) != registered_outputs_len:
        raise RuntimeError(
            f'The lengths of the meta output {len(meta_outputs)} and '
            f'the register output keys {registered_outputs_len} do not match.')

    output_nodes = []
    for i in range(registered_outputs_len):
        empty_shape = construct_fx_node_shape(meta_outputs[i].shape, sym_inputs)
        logger.debug("Construct empty out sym shape %s for fx node[name: %s][type: %s].",
                     empty_shape, node.name, node.target)
        tmp_out = graph_module.graph.call_function(torch.empty,
                                                   args=(empty_shape,),
                                                   kwargs={'dtype': meta_outputs[i].dtype,
                                                           'device': meta_outputs[i].device}
                                                   )
        output_nodes.append(tmp_out)

    # The replaced out operator have a new added kwargs input with key "out" and value tensor/list.
    if len(output_nodes) == 1:
        output_node = output_nodes[0]
        kwargs_dict["out"] = output_node
    else:
        output_node = output_nodes
        kwargs_dict["out"] = output_nodes
    return output_node


def replace_dynamic_workspace_ops(graph_module: fx.GraphModule, meta_inputs: List):
    logger.debug("Start to replace dynamic workspace ops to static workspace for ops[%s] in graph[%s].",
                 _REPLACE_FUNC_MAP.keys(), id(graph_module))

    input_node = []
    for cur_node in graph_module.graph.nodes:
        if cur_node.op == "placeholder":
            input_node.append(cur_node)
    sym_inputs_2_node = {}
    for idx, meta in enumerate(meta_inputs):
        if is_sym(meta):
            sym_inputs_2_node[meta.node.expr] = input_node[idx]

    erase_nodes = []
    for node in graph_module.graph.nodes:
        if node.target not in _REPLACE_FUNC_MAP.keys():
            continue

        with graph_module.graph.inserting_before(node):
            node_kwargs = dict(node.kwargs)
            workspace_node = construct_and_add_workspace(node, graph_module, node_kwargs)
            output_node = construct_and_add_output(node, graph_module, node_kwargs, sym_inputs_2_node)
            out_run_node = graph_module.graph.call_function(_REPLACE_FUNC_MAP[node.target].out_operator,
                                                            args=node.args, kwargs=node_kwargs)
            logger.debug("Original fx node[name: %s][type: %s] is replaced by "
                         "workspace_node[name: %s][type: %s] and run_node[name: %s][type: %s] in graph[%s].",
                         node.name, node.target,
                         workspace_node.name if workspace_node is not None else "None",
                         workspace_node.target if workspace_node is not None else "None",
                         out_run_node.name, out_run_node.target, id(graph_module))
        # replace fx
        node.replace_all_uses_with(out_run_node)
        erase_nodes.append(node)

    for node in erase_nodes:
        graph_module.graph.erase_node(node)
    graph_module.graph.lint()
    logger.debug("End to replace dynamic workspace ops in graph[%s].", id(graph_module))


def _is_inplace_node(node: Node):
    # simple analysis of function schema to determine
    # if this is an inplace variant, it might not
    # be entirely correct, but it's good enough for now.
    if isinstance(node.target, str):
        return node.target.endswith("_")
    elif hasattr(node.target, 'overloadpacket') and hasattr(node.target.overloadpacket, '__name__'):
        return node.target.overloadpacket.__name__.endswith("_")
    else:
        return False


def _find_mutated_user_inputs(gm: fx.GraphModule):
    inplace_node_args_list = []
    placeholder_args = set()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholder_args.add(node.target)
        elif _is_inplace_node(node):
            inplace_node_args_list.append(node.args[0].name)
    return [arg for arg in inplace_node_args_list if arg in placeholder_args]


@dataclass
class AclGraphCacheInfo:
    pool: Any
    stream: Any
    capture_error_mode: str
    num_warmup_iters: int
    fx_graph_name: str
    user_inputs_mapping: Dict[str, int]
    parameter_user_inputs: List[int] = None
    unupdated_sym_input_index: List[int] = None
    updated_ops_param: Dict[str, List] = None
    ops_update_rulers: Dict[str, List] = None
    mutated_user_inputs: List[str] = None
    tagged_event_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.pool is not None:
            if not (isinstance(self.pool, tuple) and len(self.pool) == 2 and
                    isinstance(self.pool[0], int) and isinstance(self.pool[1], int) and
                    self.pool[0] * self.pool[1] == 0):
                raise TypeError(f"Invalid graph pool handle type, "
                                f"got value={self.pool}, type={type(self.pool).__name__}")


class AclGraph(object):
    """
    Class representing an optimized graph for Ascend NPU execution.
    It also has the capabilities of capturing and replaying graphs.
    """

    def __init__(self, fx_graph: torch.fx.GraphModule = None, fx_forward=None, config=None):
        try:
            import torch_npu
        except ImportError as e:
            raise RuntimeError(
                "Couldn't import torch_npu. When the CompilerConfig.mode is reduce-overhead, "
                "it is necessary to use torch_npu.npu.NPUGraph(), so importing torch_npu is essential.") from e

        self._fx_forward = fx_forward
        self._fx_graph = fx_graph
        if (fx_forward is None) == (fx_graph is None):
            raise AssertionError(f"Unsupported init method: "
                                 f"must provide exactly one of either fx_forward or fx_graph.")
        self._config = config if config is not None else {}

        # members for npugraph
        self._fx_graph_name = None
        self._mempool = None
        self._stream = None
        self._capture_error_mode = None
        self._num_warmup_iters = None
        self._device = torch_npu.npu.current_device()
        self._original_mem_state = None
        self._graphs_meta: Dict[str, GraphMeta] = {}
        self.stale_storages_ptr = set()

        # members for capture, provided by AclConcreteGraph
        self._fallback_to_eager = False
        self._captured = False
        self._updated_ops_param = None
        self._unupdated_sym_input_index = None
        self._ops_update_rulers = None
        self._unupdated_input_func = None
        self._updated_input_func = None
        self._user_inputs_mapping = OrderedDict()
        self._mutated_user_inputs = None
        self._updated_node_infos = []
        self._tagged_event_names = None
        self._parameter_user_inputs = []

    @property
    def config(self):
        return self._config

    @property
    def graph(self):
        # is a Dict:[str, obj(torch_npu.npu.NPUGraph())], which mapping graph_key to NPUGraph
        return {graph_key: graph_meta.acl_graph for graph_key, graph_meta in self._graphs_meta.items()}

    @property
    def graphs_meta(self):
        return self._graphs_meta

    @property
    def name(self):
        return self._fx_graph_name

    @property
    def pool(self):
        return self._mempool

    @property
    def stream(self):
        return self._stream

    @property
    def capture_error_mode(self):
        return self._capture_error_mode

    @property
    def num_warmup_iters(self):
        return self._num_warmup_iters

    @property
    def device(self):
        return self._device

    @property
    def fx_graph(self):
        return self._fx_graph

    @property
    def fx_forward(self):
        return self._fx_forward

    @property
    def fallback_to_eager(self):
        return self._fallback_to_eager

    def fx_run_eagerly(self, *args: Any, **kwargs: Any) -> Any:
        # No need to confirm whether it is an online case or a cached case
        if self.fx_graph is not None:
            callable_fx_func = self.fx_graph
        else:
            callable_fx_func = self.fx_forward
        return callable_fx_func(*args, **kwargs)

    def load(self, aclgraph_cache_info: AclGraphCacheInfo):
        # call load before compile
        self._unupdated_sym_input_index = aclgraph_cache_info.unupdated_sym_input_index
        self._ops_update_rulers = aclgraph_cache_info.ops_update_rulers
        self._updated_ops_param = aclgraph_cache_info.updated_ops_param
        self._user_inputs_mapping = aclgraph_cache_info.user_inputs_mapping
        self._mutated_user_inputs = aclgraph_cache_info.mutated_user_inputs
        self._parameter_user_inputs = aclgraph_cache_info.parameter_user_inputs
        self._mempool = aclgraph_cache_info.pool if aclgraph_cache_info.pool is not None else \
            torch.npu.graph_pool_handle()
        self._stream = aclgraph_cache_info.stream if aclgraph_cache_info.stream is not None else torch.npu.Stream()
        self._capture_error_mode = aclgraph_cache_info.capture_error_mode
        self._num_warmup_iters = aclgraph_cache_info.num_warmup_iters
        self._fx_graph_name = aclgraph_cache_info.fx_graph_name
        self._tagged_event_names = aclgraph_cache_info.tagged_event_names

    def compile(self, *args: Any, **kwargs: Any):
        if not self._captured:
            # warm up before capture
            with record_function("acl_graph_warm_up"):
                for _ in range(self.num_warmup_iters):
                    self.fx_run_eagerly(*args, **kwargs)
                    torch.npu.synchronize()

            # compile operator kernel based on static shape for better execution performance
            if self.config.get('_aclnn_static_shape_kernel', False):
                path = self.config.get('_aclnn_static_shape_kernel_build_dir', None)
                fx_func = self.fx_graph if self.fx_graph is not None else self.fx_forward
                compile_static_kernel(fx_func, *args, build_dir=path, **kwargs)

            self._unupdated_input_func = get_unupdated_input_fn(self._unupdated_sym_input_index)
            self._updated_input_func = get_updated_ops_fn(self._ops_update_rulers)
            self._captured = True

            # In the current version, the initialization of mem pool requires an explicit call to capture.
            # In versions greater than 2.6, the initialization can be completed directly when creating the mem pool.
            import torch_npu
            s = torch_npu.npu.Stream()
            with torch_npu.npu.stream(s):
                g = torch_npu.npu.NPUGraph()
                g.capture_begin(pool=self.pool)
                g.capture_end()
            # record the original memory state before capture,
            # and it will be used to restore the mem state when capturing another acl graph for different shape.
            self._original_mem_state = torch_npu._C._npu_getCheckpointState(self.device, self.pool)

        if self.fallback_to_eager:
            # when falling back to eager, no need to calculate graph key
            return "fallback_to_eager"

        saved_graph_meta = None
        # get graph key based on unupdated sym input shape or value
        graph_key = self._unupdated_input_func(*args, **kwargs)
        if graph_key in self._graphs_meta.keys():
            logger.info('Find captured AclGraph{id: %s} of fx_graph %s with graph key {%s}.',
                        id(self.graph[graph_key]), self.name, graph_key)
            if self.is_need_to_recapture(graph_key, *args):
                logger.debug('The current AclGraph needs to be recaptured for fx_graph %s with graph key {%s}.',
                             self.name, graph_key)
                # save graph_meta, release resources after recapture
                saved_graph_meta = self._graphs_meta[graph_key]
                del self._graphs_meta[graph_key]
            else:
                logger.debug('The current AclGraph no needs to be recaptured for fx_graph %s with graph key {%s}.',
                             self.name, graph_key)
                return graph_key

        # Before recapture a new aclgraph for another graph key, check static capture size limit.
        # If the number of captured aclgraphs exceeds the limit, we will fall back to eager execution.
        if len(self._graphs_meta) == int(self.config.get("static_capture_size_limit", "-1")):
            warn_msg = f"The static_capture_size_limit reached when capturing fx_graph {self.name} " \
                       f"with graph key {graph_key}, we will fall back to eager for all subsequent executions. " \
                       f"Excessive recapture can degrade performance due to the each recapture overhead, " \
                       f"and result in unexpected resources consumption, including stream, memory, etc. " \
                       f"This may lead to program error: The resources are insufficient. " \
                       f"The current static_capture_size_limit " \
                       f"is {self.config.get('static_capture_size_limit', '-1')}. If recapture are expected, " \
                       f"consider increasing debug.aclgraph.static_capture_size_limit to an appropriate value."
            warnings.warn(warn_msg)
            logger.warning(warn_msg)
            self._fallback_to_eager = True
            self.reset_captured_graph()
            return "fallback_to_eager"

        # Start capture aclgraph instance when the graph key have not been compiled.
        self.compile_for_graph_key(graph_key, *args, **kwargs)
        if saved_graph_meta is not None:
            saved_graph_meta.acl_graph.reset()

        return graph_key

    def reset_captured_graph(self):
        # Do sync before we reset all captured aclgraphs, just like what was done before capture
        torch.npu.synchronize()
        logger.info('Current fx_graph %s memory pool is %s. Before reset, the current memory state is {%s}.',
                    self.name, self.pool, debug_mem_state())

        for _, graph_meta in self._graphs_meta.items():
            graph_meta.acl_graph.reset()
        self._graphs_meta = {}
        gc.collect()
        torch.npu.empty_cache()

        logger.info('Current fx_graph %s memory pool is %s. After reset, the current memory state is {%s}.',
                    self.name, self.pool, debug_mem_state())

    def compile_for_graph_key(self, graph_key, *args: Any, **kwargs: Any):
        """
        Compile the FX graph into another new ACL graph instance with specific graph key.

        Args:
            graph_key (str): Unique identifier for the captured ACL graph.
            *args: Input arguments for FX graph.
            **kwargs: Keyword arguments for FX graph.
        """

        import torch_npu
        self._graphs_meta[graph_key] = GraphMeta(graph_key=graph_key,
                                                 acl_graph=torch_npu.npu.NPUGraph(),
                                                 replay_func=None,
                                                 captured_inputs=args,
                                                 outputs_meta=[],
                                                 outputs_weakref=[],
                                                 mem_state_after_capture=None,
                                                 captured_parameter={})
        # record the parameter address for subsequent comparison to determine if recapture is necessary
        for parameter_idx in self._parameter_user_inputs:
            self.graphs_meta[graph_key].captured_parameter.setdefault(parameter_idx, args[parameter_idx].data_ptr())
        logger.debug('No find captured AclGraph{id: %s} of fx_graph %s with graph key {%s}, and start to capture it.',
                     id(self.graph[graph_key]), self.name, graph_key)

        # get stale storages from last run for current fx graph
        stale_storage_set = set()
        for key, _ in self._graphs_meta.items():
            if self._graphs_meta[key].outputs_weakref is None:
                continue
            for output_ref in self._graphs_meta[key].outputs_weakref:
                ref = output_ref()
                if ref is not None and isinstance(ref, torch.Tensor) and \
                        ref.untyped_storage()._cdata in self.stale_storages_ptr:
                    stale_storage_set.add(ref.untyped_storage()._cdata)
        stale_storages = list(stale_storage_set)

        # set to common original memory state before capture
        enable_mempool_reuse = not ("disable_mempool_reuse_in_same_fx" in self.config.keys() and self.config[
            "disable_mempool_reuse_in_same_fx"] == "1")
        if enable_mempool_reuse:
            logger.debug('Start setting to original memory state for fx_graph %s with graph key{%s}. '
                         'The stale storage is %s, and the current memory state is {%s}.',
                         self.name, graph_key, stale_storages, debug_mem_state())
            torch_npu._C._npu_setCheckpointPoolState(self.device, self._original_mem_state, stale_storages, [])
            logger.debug('After setting to original memory state for fx_graph %s with graph key{%s}. '
                         'The current memory state is {%s}.',
                         self.name, graph_key, debug_mem_state())
            for ptr in stale_storages:
                self.stale_storages_ptr.discard(ptr)

        # start capture aclgraph
        with record_function("acl_graph_capture"):
            captured_outputs = self.capture(graph_key, *args, **kwargs)

        # The captured output tensors will not be held indefinitely and its will be terminated after this capture.
        self._graphs_meta[graph_key].mem_state_after_capture = torch_npu._C._npu_getCheckpointState(self.device,
                                                                                                    self.pool)
        logger.debug('After capturing fx_graph %s for graph key{%s} to AclGraph{id: %s}, '
                     'memory pool reuse is %s, the memory state is {%s}.',
                     self.name, graph_key, id(self.graph[graph_key]),
                     "enable" if enable_mempool_reuse else "disable", debug_mem_state())

        if enable_mempool_reuse:
            del captured_outputs
        else:
            self._graphs_meta[graph_key].retained_outputs = captured_outputs

    def capture(self, graph_key, *args: Any, **kwargs: Any):
        """
        Captures the execution of the FX graph into an ACL graph instance.
        
        Args:
            graph_key (str): Unique identifier for the captured graph.
            *args: Input arguments for graph capture.
            **kwargs: Keyword arguments for graph capture.
        """
        sync_launch_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        if sync_launch_env == "1":
            raise RuntimeError(f"Stream synchronization is unsupported when capturing task for AclGraph. "
                               f"Please unset ASCEND_LAUNCH_BLOCKING env variable before capture.")

        if self.fx_graph is not None:
            captured_interpreter = UpdatedNodeCaptureInterp(self.fx_graph, self._updated_ops_param)
            import torch_npu
            with torch_npu.npu.graph(self.graph[graph_key], pool=self.pool, stream=self.stream,
                                     capture_error_mode=self.capture_error_mode):
                captured_outputs = captured_interpreter.run(*args, **kwargs)
                self._updated_node_infos = captured_interpreter.captured_node_infos

        else:
            import torch_npu
            with torch_npu.npu.graph(self.graph[graph_key], pool=self.pool, stream=self.stream,
                                     capture_error_mode=self.capture_error_mode):
                captured_outputs = self.fx_forward(*args, node_info=self._updated_node_infos, **kwargs)
                for i, _ in enumerate(self._updated_node_infos):
                    logger.debug("Record the %s th updated node, node name[%s], node func[%s], node args length[%s], "
                                 "node kwargs length[%s], update param name[%s], update task handle[%s], "
                                 "update event[%s] in graph.",
                                 i,
                                 self._updated_node_infos[i].node_name,
                                 self._updated_node_infos[i].updated_func,
                                 len(self._updated_node_infos[i].args),
                                 len(self._updated_node_infos[i].kwargs),
                                 self._updated_node_infos[i].updated_param_name,
                                 self._updated_node_infos[i].handle,
                                 self._updated_node_infos[i].event
                                 )

        logger.info('Success to capture fx_graph %s for graph key{%s}. '
                    'Start to run AclGraph{id: %s} with the updated node num {%s}.',
                    self.name, graph_key, id(self.graph[graph_key]), len(self._updated_node_infos))

        # save outputs meta info and weakref
        for output_iter in captured_outputs:
            if isinstance(output_iter, torch.Tensor):
                weak_ref = WeakRef(None)
            else:
                weak_ref = WeakRef(output_iter)
            self._graphs_meta[graph_key].outputs_weakref.append(weak_ref)
            self._graphs_meta[graph_key].outputs_meta.append(get_tensor_metadata(output_iter))
        logger.debug('In fx_graph %s, all the non parameter tensor inputs are %s. all the output meta info are %s.',
                     self.name, self._user_inputs_mapping, self._graphs_meta[graph_key].outputs_meta)

        # gen run func
        self._graphs_meta[graph_key].replay_func = CapturedGraphUpdateAndReplay(self.graph[graph_key],
                                                                                self._updated_input_func,
                                                                                self._updated_node_infos)

        return captured_outputs

    def process_input(self, graph_key, *args: Any):
        for idx in self._user_inputs_mapping.values():
            if self.graphs_meta[graph_key].captured_inputs[idx].data_ptr() != args[idx].data_ptr():
                self.graphs_meta[graph_key].captured_inputs[idx].copy_(args[idx])

    def run(self, graph_key, *args, **kwargs):
        self._graphs_meta[graph_key].replay_func(*args, **kwargs)
    
    def is_need_to_recapture(self, graph_key, *args: Any):
        enable_parameter_frozen = "frozen_parameter" in self.config.keys() \
                                  and self.config["frozen_parameter"] == "1"
        # When the memory addresses of mutated_inputs and parameter type inputs change
        # recapture the aclgraph to reduce copy time and improve performance
        for input_name, input_idx in self._user_inputs_mapping.items():
            if self.graphs_meta[graph_key].captured_inputs[input_idx].data_ptr() != args[input_idx].data_ptr():
                if input_name in self._mutated_user_inputs:
                    return True
        
        # Check if the parameter address has changed
        if not enable_parameter_frozen:
            for idx, parameter_ptr in self._graphs_meta[graph_key].captured_parameter.items():
                if parameter_ptr != args[idx].data_ptr():
                    return True
        return False

    def reconstruct_outputs(self, graph_key: str) -> List:
        """
        Reconstruct output tensors according to their saved metadata.
        Do not increase the reference count to the output tensors, and only weak reference is recorded.
        """

        disable_reuse = "disable_mempool_reuse_in_same_fx" in self.config.keys() \
                        and self.config["disable_mempool_reuse_in_same_fx"] == "1"
        if disable_reuse:
            # When no mempool reuse in same fx, the retained outputs no need to reconstruct.
            logger.debug('When config.debug.aclgraph.disable_mempool_reuse_in_same_fx is True, '
                         'no mempool reuse in fx_graph %s for graph key{%s}, all the outputs are retained.',
                         self.name, graph_key)
            return self._graphs_meta[graph_key].retained_outputs

        if len(self._graphs_meta[graph_key].outputs_meta) != len(self._graphs_meta[graph_key].outputs_weakref):
            raise RuntimeError(
                f'The lengths of the outputs tensor meta {len(self._graphs_meta[graph_key].outputs_meta)} and '
                f'the outputs tensor ref {len(self._graphs_meta[graph_key].outputs_weakref)} do not match.')

        # for case: all output storages are alive.
        ret = self.construct_outputs_based_on_ref(graph_key)
        if ret is not None:
            logger.debug('All output tensors weak ref are valid, '
                         'no need to reconstruct output tensors for fx_graph %s with graph key{%s}.',
                         self.name, graph_key)
            return ret

        # reconstructing step 1: set alive output of last execution to stale storage.
        self.set_to_original_state_bofore_reconstruct(graph_key)

        # reconstructing step 2: reconstruct output tensor based on tensor meta info.
        outputs = []
        for idx, output_meta in enumerate(self._graphs_meta[graph_key].outputs_meta):
            output_ref = self._graphs_meta[graph_key].outputs_weakref[idx]()
            if output_ref is None or isinstance(output_ref, torch.Tensor):
                # Reconstruct outputs for the following case:
                # 1. Invalid tensor ref.
                # 2. Valid tensor ref that have been set to stale storage.
                output_i = reconstruct_from_tensor_metadata(output_meta)
                self._graphs_meta[graph_key].outputs_weakref[idx].swap_weakref(output_i)
                outputs.append(output_i)
            else:
                # other non tensor obj can be returned directly.
                outputs.append(output_ref)

        # reconstructing step 3: Associate output tensor and data_ptr by setting deleter or clone.
        enable_output_clone = "enable_output_clone" in self.config.keys() and self.config[
            "enable_output_clone"] == "1"
        if enable_output_clone:
            outputs = [out.clone() if isinstance(out, torch.Tensor) else out for out in outputs]
        else:
            self.set_reconstructed_outputs_deleter(graph_key, outputs)
            warn_msg = "Because acl graph fixes memory addresses, acl graphs do not have a great way of " \
                       "handling live tensors from a previous invocation. " \
                       "The retained memory of acl graph output tensors will be overwritten by " \
                       "subsequent executions, which may cause precision issues. " \
                       "See https://docs.pytorch.org/docs/main/torch.compiler_cudagraph_trees.html#limitations " \
                       "for more details. To resolve this, you can either: " \
                       "1. Remove the memory retention of all the output tensors in your script; " \
                       "2. Enable debug option by setting debug.aclgraph.enable_output_clone=True."
            warnings.warn(warn_msg)
            warnings.filterwarnings("ignore", message=warn_msg)

        return outputs

    def construct_outputs_based_on_ref(self, graph_key: str) -> List:
        """
        No need to set deleter for those reconstructed output tensor when weak reference is not None.

        Args:
            graph_key (str): Unique identifier for the captured ACL graph.
        """

        returned_outputs = []
        alive_outputs = {}
        reconstructed_outputs = {}
        for idx, output_meta in enumerate(self._graphs_meta[graph_key].outputs_meta):
            output_ref = self._graphs_meta[graph_key].outputs_weakref[idx]()
            if output_ref is not None:
                returned_outputs.append(output_ref)
                if isinstance(output_ref, torch.Tensor):
                    # record alive output ptr and storage map
                    alive_outputs[output_ref.untyped_storage().data_ptr()] = output_ref
            else:
                # reconstruct output that need to set storage.
                output_i = reconstruct_from_tensor_metadata(output_meta)
                returned_outputs.append(output_i)
                if output_i.untyped_storage().data_ptr() not in reconstructed_outputs.keys():
                    reconstructed_outputs[output_i.untyped_storage().data_ptr()] = [output_i]
                else:
                    reconstructed_outputs[output_i.untyped_storage().data_ptr()].append(output_i)

        for ptr, tensor_list in reconstructed_outputs.items():
            if ptr not in alive_outputs.keys():
                # when reconstructed output ptr is not in alive output, setting deleter is necessary.
                return None
            for tensor_i in tensor_list:
                # set output storage for those shared memory output
                tensor_i.set_(alive_outputs[ptr].untyped_storage())
        return returned_outputs

    def set_to_original_state_bofore_reconstruct(self, graph_key: str) -> None:
        """
        In the case of memory reuse for aclgraphs,
        in order to set the memory state for the reconstructed outputs,
        it is necessary to set those alive output tensors of other graph keys to stale
        and reset the memory pool state to its original state before capture.

        Args:
            graph_key (str): Unique identifier for the captured ACL graph.
        """

        # Graph outputs may be freed by user, we just get tensor that still alive after last execution.
        stale_storage_set = set()
        for key, _ in self._graphs_meta.items():
            for output_ref in self._graphs_meta[key].outputs_weakref:
                ref = output_ref()
                if ref is not None and isinstance(ref, torch.Tensor) and \
                        ref.untyped_storage()._cdata in self.stale_storages_ptr:
                    stale_storage_set.add(ref.untyped_storage()._cdata)
        other_graph_stale_storages = list(stale_storage_set)

        import torch_npu
        if len(other_graph_stale_storages) > 0:
            logger.debug('Before reset fx_graph %s outputs stale storages cdata %s to original memory state '
                         'for AclGraph{id: %s} with the current graph key{%s}, and the current memory state is {%s}.',
                         self.name, other_graph_stale_storages, id(self.graph[graph_key]), graph_key, debug_mem_state())
            # Reset other graph live tensors to stale storages
            torch_npu._C._npu_setCheckpointPoolState(self.device, self._original_mem_state,
                                                     other_graph_stale_storages, [])
            self.stale_storages_ptr = set()

        logger.debug('After reset fx_graph %s outputs stale storages, '
                     'for AclGraph{id: %s} with the current graph key{%s}, and the current memory state is {%s}.',
                     self.name, id(self.graph[graph_key]), graph_key, debug_mem_state())

    def set_reconstructed_outputs_deleter(self, graph_key: str, reconstructed_outputs: List[torch.Tensor]) -> None:
        """
        The output tensor is reconstructed based on tensor meta info.
        This tensor data_ptr is only a bare pointer and is not associated with memory blocks.
        So we need to set check point pool state and add deleter Fn.
        After that, the right memory block state can be returned by memory_snapshot.

        Args:
            graph_key (str): Unique identifier for the captured ACL graph.
            reconstructed_outputs (List[torch.Tensor]): The reconstructed outputs that need to set deleter.
        """

        # Just get output storages by unique storage data ptr.
        # Attempt to handle cases where multiple outputs are shared memory.
        # Eg 'return x, x.view(-1), x[0]'.
        # In this case, these three output tensors should be constructed based on the same storage
        all_reconstructed_storages_ptr = {}
        reconstructed_outputs_to_add_deleter = []
        for output_i in reconstructed_outputs:
            if isinstance(output_i, torch.Tensor):
                if output_i.untyped_storage().data_ptr() not in all_reconstructed_storages_ptr.keys():
                    reconstructed_outputs_to_add_deleter.append(output_i.untyped_storage()._cdata)
                    all_reconstructed_storages_ptr[output_i.untyped_storage().data_ptr()] = [output_i]
                else:
                    all_reconstructed_storages_ptr[output_i.untyped_storage().data_ptr()].append(output_i)

        # Currently we deallocate on instead of allowing stale recordings
        stale_storages: List[int] = []
        import torch_npu
        logger.debug('Before reconstructing fx_graph %s outputs for graph key{%s}, '
                     'the storages to add deleter are %s, the memory state is {%s}.',
                     self.name, graph_key, reconstructed_outputs_to_add_deleter, debug_mem_state())
        # Set reconstructed outputs deleter fn
        torch_npu._C._npu_setCheckpointPoolState(self.device, self._graphs_meta[graph_key].mem_state_after_capture,
                                                 stale_storages, reconstructed_outputs_to_add_deleter)

        # When multiple Python tensors have the same storage ptr, they should have the same storage object
        for _, tensor_list in all_reconstructed_storages_ptr.items():
            if len(tensor_list) < 2:
                continue
            for tensor_i in tensor_list[1:]:
                tensor_i.set_(tensor_list[0].untyped_storage())

        self.stale_storages_ptr.update(reconstructed_outputs_to_add_deleter)
        logger.debug('After reconstructing fx_graph %s outputs for graph key{%s}, '
                     'the memory state is {%s}.',
                     self.name, graph_key, debug_mem_state())
