from collections import deque, OrderedDict
from typing import List, Optional, Callable, Any, Deque, Dict, Set, Tuple, Union
from dataclasses import dataclass
import functools
import gc
import os
import pickle
import sys
import sympy

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


@dataclass
class StaticWorkspaceReplaceFunc:
    get_workspace: Callable
    out_operator: Callable
    workspace_keys: List[str]
    output_keys: List[str]
    updated_param_keys: List[str]


@dataclass
class UpdatedNodeInfo:
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
    torch.ops.npu.npu_fused_infer_attention_score_v2.default: StaticWorkspaceReplaceFunc(
        get_workspace=torch.ops.npu._npu_fused_infer_attention_score_v2_get_max_workspace.default,
        out_operator=torch.ops.npu.npu_fused_infer_attention_score_v2.out,
        workspace_keys=["workspace"],
        output_keys=["attention_out", "softmax_lse"],
        updated_param_keys=["actual_seq_qlen", "actual_seq_kvlen"],
    ),
}


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
    user_inputs_mapping: Dict[str, List]
    unupdated_sym_input_index: List[int] = None
    updated_ops_param: Dict[str, List] = None
    ops_update_rulers: Dict[str, List] = None
    mutated_user_inputs: List[str] = None


class AclGraph(object):
    def __init__(self, fx_graph: torch.fx.GraphModule = None, serialized_fx_graph=None, config=None):
        try:
            import torch_npu
        except ImportError as e:
            raise RuntimeError(
                "Couldn't import torch_npu. When the CompilerConfig.mode is reduce-overhead, "
                "it is necessary to use torch_npu.npu.NPUGraph(), so importing torch_npu is essential.") from e

        if fx_graph is not None and serialized_fx_graph is None:
            self._fx_graph = fx_graph
        elif fx_graph is None and serialized_fx_graph is not None:
            self._fx_graph = self.load_graphmodule_from_str(serialized_fx_graph)
        else:
            raise AssertionError(f"Unsupported init method: "
                                 f"must provide exactly one of either fx_graph or serialized_fx_graph.")
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

        # members for capture, provided by AclConcreteGraph
        self._captured = False
        self._updated_ops_param = None
        self._unupdated_sym_input_index = None
        self._ops_update_rulers = None
        self._unupdated_input_func = None
        self._updated_input_func = None
        self._user_inputs_mapping = OrderedDict()
        self._mutated_user_inputs = None

    @property
    def config(self):
        return self._config

    @property
    def graph(self):
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

    @staticmethod
    def save_graphmodule_to_str(gm: torch.fx.GraphModule):
        serialized_to_str = ""

        if isinstance(gm, torch.fx.GraphModule):
            reduce_data = gm.__reduce__()
            data = (reduce_data, gm.state_dict())
            try:
                serialized_to_str = pickle.dumps(data)
            except Exception as e:
                logger.warning(f"Faild to serialize fx graph, error msg: {e}")

        return serialized_to_str

    def load(self, aclgraph_cache_info: AclGraphCacheInfo):
        # call load before compile
        self._unupdated_sym_input_index = aclgraph_cache_info.unupdated_sym_input_index
        self._ops_update_rulers = aclgraph_cache_info.ops_update_rulers
        self._updated_ops_param = aclgraph_cache_info.updated_ops_param
        self._user_inputs_mapping = aclgraph_cache_info.user_inputs_mapping
        self._mutated_user_inputs = aclgraph_cache_info.mutated_user_inputs
        self._mempool = aclgraph_cache_info.pool if aclgraph_cache_info.pool is not None else \
            torch.npu.graph_pool_handle()
        self._stream = aclgraph_cache_info.stream if aclgraph_cache_info.stream is not None else torch.npu.Stream()
        self._capture_error_mode = aclgraph_cache_info.capture_error_mode
        self._num_warmup_iters = aclgraph_cache_info.num_warmup_iters
        self._fx_graph_name = aclgraph_cache_info.fx_graph_name

    def compile(self, *args: Any, **kwargs: Any):
        if not self._captured:
            # warm up before capture
            with record_function("acl_graph_warm_up"):
                for _ in range(self.num_warmup_iters):
                    self.fx_graph(*args, **kwargs)
                    torch.npu.synchronize()

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

        # get graph key based on unupdated sym input shape or value
        graph_key = self._unupdated_input_func(*args, **kwargs)
        if graph_key in self._graphs_meta.keys():
            logger.debug('Find captured AclGraph{id: %s} of fx_graph %s with graph key {%s}.',
                         id(self.graph[graph_key]), self.name, graph_key)
            return graph_key

        # start capture aclgraph
        import torch_npu
        self._graphs_meta[graph_key] = GraphMeta(graph_key=graph_key,
                                                 acl_graph=torch_npu.npu.NPUGraph(),
                                                 replay_func=None,
                                                 captured_inputs=args,
                                                 outputs_meta=[],
                                                 outputs_weakref=[],
                                                 mem_state_after_capture=None,
                                                 is_first_replay=True)
        logger.debug('No find captured AclGraph{id: %s} of fx_graph %s with graph key {%s}, and start to capture it.',
                     id(self.graph[graph_key]), self.name, graph_key)

        stale_storage_set = set()
        for key, _ in self._graphs_meta.items():
            if self._graphs_meta[key].outputs_weakref is None:
                continue
            for output_ref in self._graphs_meta[key].outputs_weakref:
                ref = output_ref()
                if ref is not None and isinstance(ref, torch.Tensor):
                    stale_storage_set.add(ref.untyped_storage()._cdata)
        stale_storages = list(stale_storage_set)

        enable_mempool_reuse = not ("disable_mempool_reuse_in_same_fx" in self.config.keys() and self.config[
            "disable_mempool_reuse_in_same_fx"] == "1")
        enable_output_clone = "enable_output_clone" in self.config.keys() and self.config[
            "enable_output_clone"] == "1"
        if enable_mempool_reuse and enable_output_clone:
            torch_npu._C._npu_setCheckpointPoolState(self.device, self._original_mem_state, stale_storages, [])
            logger.debug('After setting to original memory state for fx_graph %s for graph key{%s}. '
                         'The stale storage is %s, and the current memory state is {%s}.',
                         self.name, graph_key, stale_storages, debug_mem_state())

        # start capture
        with record_function("acl_graph_capture"):
            self.capture(graph_key, *args, **kwargs)

        return graph_key

    def capture(self, graph_key, *args: Any, **kwargs: Any):
        captured_interpreter = UpdatedNodeCaptureInterp(self.fx_graph, self._updated_ops_param)

        import torch_npu
        with torch_npu.npu.graph(self.graph[graph_key], pool=self.pool, stream=self.stream,
                                 capture_error_mode=self.capture_error_mode):
            captured_outputs = captured_interpreter.run(*args, **kwargs)

        updated_node_infos = captured_interpreter.captured_node_infos
        logger.info('Success to capture fx_graph %s for graph key{%s}. '
                    'Start to run AclGraph{id: %s} with the updated node num {%s}.',
                    self.name, graph_key, id(self.graph[graph_key]), len(updated_node_infos))

        # The captured output tensors will not be held indefinitely,
        # and its will be terminated after the capture ends.
        import torch_npu
        self._graphs_meta[graph_key].mem_state_after_capture = \
            torch_npu._C._npu_getCheckpointState(self.device, self.pool)

        for output_iter in captured_outputs:
            if isinstance(output_iter, torch.Tensor):
                weak_ref = WeakRef(None)
            else:
                weak_ref = WeakRef(output_iter)
            self._graphs_meta[graph_key].outputs_weakref.append(weak_ref)
            self._graphs_meta[graph_key].outputs_meta.append(get_tensor_metadata(output_iter))
        logger.debug('After capturing fx_graph %s for graph key{%s} to AclGraph{id: %s}, the memory state is {%s}.',
                     self.name, graph_key, id(self.graph[graph_key]), debug_mem_state())

        disable_reuse = "disable_mempool_reuse_in_same_fx" in self.config.keys() \
                        and self.config["disable_mempool_reuse_in_same_fx"] == "1"
        if disable_reuse:
            self._graphs_meta[graph_key].retained_outputs = captured_outputs
        else:
            del captured_outputs
        logger.debug('In fx_graph %s, memory pool reuse state is %s in same fx graph, '
                     'and all the non parameter tensor input is %s.',
                     self.name, "disable" if disable_reuse else "enable", self._user_inputs_mapping)

        # gen run func
        self._graphs_meta[graph_key].replay_func = CapturedGraphUpdateAndReplay(self.graph[graph_key],
                                                                                self._updated_input_func,
                                                                                updated_node_infos)

    def process_input(self, graph_key, *args: Any):
        for idx in self._user_inputs_mapping.values():
            if self.graphs_meta[graph_key].captured_inputs[idx].data_ptr() != args[idx].data_ptr():
                self.graphs_meta[graph_key].captured_inputs[idx].copy_(args[idx])

    def run(self, graph_key, *args, **kwargs):
        self._graphs_meta[graph_key].replay_func(*args, **kwargs)

    def process_inplace_inputs(self, graph_key, *args: Any):
        # For in-place op, dynamo will transform it into a functionalized call and add copy_ node when setting
        # keep_inference_input_mutations=True, which may need data copy from capture input to user input (when tensor
        # address is different between capture and replay).
        for arg_name in self._mutated_user_inputs:
            if arg_name not in self._user_inputs_mapping:
                raise RuntimeError(f"{arg_name} is not in input args: {self._user_inputs_mapping.keys()}")
            idx = self._user_inputs_mapping[arg_name]
            if self.graphs_meta[graph_key].captured_inputs[idx].data_ptr() != args[idx].data_ptr():
                logger.warning_once(f"Mutated input[{arg_name}]'s data_ptr is different between capture and replay. "
                                    f"This may call redundant copy. ")
                args[idx].copy_(self.graphs_meta[graph_key].captured_inputs[idx])

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

        outputs = []
        have_invalid_weakref = False

        for idx, output_meta in enumerate(self._graphs_meta[graph_key].outputs_meta):
            output_ref = self._graphs_meta[graph_key].outputs_weakref[idx]()
            if output_ref is None:
                output_i = reconstruct_from_tensor_metadata(output_meta)
                self._graphs_meta[graph_key].outputs_weakref[idx].swap_weakref(output_i)
                outputs.append(output_i)
                have_invalid_weakref = True
            else:
                # valid tensor ref and other type obj can be returned directly.
                outputs.append(output_ref)

        if have_invalid_weakref:
            enable_output_clone = "enable_output_clone" in self.config.keys() and self.config[
                "enable_output_clone"] == "1"
            if enable_output_clone:
                self.set_reconstructed_outputs_deleter(graph_key, outputs)
                # TO DO: Add clone temporarily. Maybe no deleter in the future.
                clone_outputs = []
                for out in outputs:
                    if isinstance(out, torch.Tensor):
                        clone_outputs.append(out.clone())
                    else:
                        clone_outputs.append(out)
                outputs = clone_outputs
            else:
                logger.warning_once(
                    f"When debug.aclgraph.enable_output_clone is False, output tensor should not be retained, "
                    f"otherwise it may cause functional errors for mempool reuse in same fx graph.")
        else:
            logger.debug('All output tensors weak ref are valid, '
                         'no need to reconstruct fx_graph %s for graph key{%s}.',
                         self.name, graph_key)

        if self._graphs_meta[graph_key].is_first_replay:
            self._graphs_meta[graph_key].is_first_replay = False

        return outputs

    def set_to_original_state_bofore_reconstruct(self, graph_key: str) -> None:
        stale_storage_set = set()
        for key, _ in self._graphs_meta.items():
            if key == graph_key:
                continue
            for output_ref in self._graphs_meta[key].outputs_weakref:
                ref = output_ref()
                if ref is not None and isinstance(ref, torch.Tensor):
                    stale_storage_set.add(ref.untyped_storage()._cdata)
        other_graph_stale_storages = list(stale_storage_set)

        import torch_npu
        if len(other_graph_stale_storages) > 0 and not self._graphs_meta[graph_key].is_first_replay:
            # reset other graph live tensors to stale storages
            torch_npu._C._npu_setCheckpointPoolState(self.device, self._original_mem_state,
                                                     other_graph_stale_storages, [])

        logger.debug('Reset fx_graph %s other graph key outputs stale storage cdata %s, '
                     'and set to original memory state for AclGraph{id: %s} with graph key{%s}.',
                     self.name, other_graph_stale_storages, id(self.graph[graph_key]), graph_key)

    def set_reconstructed_outputs_deleter(self, graph_key: str, reconstructed_outputs: List[torch.Tensor]) -> None:
        self.set_to_original_state_bofore_reconstruct(graph_key)

        reconstructed_outputs_to_add_deleter = []
        for output_i in reconstructed_outputs:
            if isinstance(output_i, torch.Tensor):
                reconstructed_outputs_to_add_deleter.append(output_i.untyped_storage()._cdata)

        # currently we deallocate on instead of allowing stale recordings
        stale_storages: List[int] = []
        import torch_npu
        torch_npu._C._npu_setCheckpointPoolState(self.device, self._graphs_meta[graph_key].mem_state_after_capture,
                                                 stale_storages, reconstructed_outputs_to_add_deleter)
        logger.debug('After reconstructing fx_graph %s graph key{%s} outputs, '
                     'the storages to add deleter are %s, the memory state is {%s}.',
                     self.name, graph_key, reconstructed_outputs_to_add_deleter, debug_mem_state())

    def load_graphmodule_from_str(self, serialized_str: str):
        reduce_data, state_dict = pickle.loads(serialized_str)
        callable_fn, args = reduce_data
        gm = callable_fn(*args)
        gm.load_state_dict(state_dict)
        gm.recompile()

        return gm
