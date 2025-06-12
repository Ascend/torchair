from collections import deque
from typing import List, Optional, Callable, Any, Deque, Dict, Set, Tuple, Union
from dataclasses import dataclass
import functools
import gc
import os
import sys
import sympy

import torch
from torch import fx
from torch import nn
from torch.fx import Node, Proxy

from torchair.core.utils import logger


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


def get_unupdated_input_fn(graph_module: torch.fx.GraphModule):
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


def get_updated_ops_fn(graph_module: torch.fx.GraphModule, meta_inputs: List):
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
    return gen_updated_input_func(ops_update_rulers), need_updated_ops


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
        output_node = graph_module.graph.call_function(tuple, args=(output_nodes,))
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
        node.replace_all_uses_with(output_node)
        erase_nodes.append(node)

    for node in erase_nodes:
        graph_module.graph.erase_node(node)
    graph_module.recompile()
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


try:
    import torch_npu
except ImportError as e:
    raise RuntimeError(
        "Couldn't import torch_npu. When the CompilerConfig.mode is reduce-overhead, "
        "it is necessary to use torch_npu.npu.NPUGraph(), so importing torch_npu is essential.") from e


class CapturedGraph(torch_npu.npu.graph):
    def __init__(self, *args, **kwargs):
        self.avoid_sync_before_capture = self._is_external_event_in_graph(kwargs["fx_graph"])
        kwargs.pop("fx_graph", None)
        super().__init__(*args, **kwargs)

    # TODOO: Mitigation: Avoid device synchronization while there is a replay interaction between two graphs.
    # graph A: op1 -> record1 -> op2 -> op3 -> wait2 -> op4
    # graph B: op5 -> op6 -> wait1 -> op7 -> op8 -> record2
    # The graph A and graph B will be executed in parallel, while capture A then replay A, then capture B and replay B.
    # if do synchronize device before capture, graph A replay is blocking due to record2 in graph B is not executed,
    # graph B is not launched right now, so rewrite torch.npu.graph.__enter__() to avoid synchronize.
    def __enter__(self):
        if not self.avoid_sync_before_capture:
            super().__enter__()
        else:
            logger.debug("Do not synchronization device to avoid graph capture is blocking.")
            gc.collect()
            self.stream_ctx.__enter__()

            self.npu_graph.capture_begin(
                *self.pool, capture_error_mode=self.capture_error_mode
            )

    @staticmethod
    def _is_external_event_in_graph(gm: fx.GraphModule):
        if gm is None:
            return False
        for node in gm.graph.nodes:
            if str(node.target) in ["air.external_event_record.default",
                                    "air.external_event_wait.default",
                                    "air.external_event_reset.default"]:
                return True
        return False
