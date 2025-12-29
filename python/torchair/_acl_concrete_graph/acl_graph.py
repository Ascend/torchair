from collections import deque, OrderedDict
from typing import List, Optional, Callable, Any, Deque, Dict, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import functools
import itertools
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
from torch.fx.node import Argument, Target
from torch.profiler import record_function

from torchair.core.utils import logger
from torchair.scope._scope_attr import guard_with_user_stream_scope
from torchair._utils.graph_transform_observer import DebugContext
from torchair._acl_concrete_graph.utils import reconstruct_args_kwargs, timer, is_inputs_base_format
from torchair._acl_concrete_graph.utils import (debug_mem_state, LazyMessage, WeakRef, GraphMeta, get_tensor_metadata,
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


_REPLACE_FUNC_MAP = {}

if hasattr(torch.ops.npu, "npu_fused_infer_attention_score"):
    _REPLACE_FUNC_MAP.update({torch.ops.npu.npu_fused_infer_attention_score.default:
        StaticWorkspaceReplaceFunc(
            get_workspace=torch.ops.npu._npu_fused_infer_attention_score_get_max_workspace.default,
            out_operator=torch.ops.npu.npu_fused_infer_attention_score.out,
            workspace_keys=["workspace"],
            output_keys=["attention_out", "softmax_lse"],
            updated_param_keys=["actual_seq_lengths", "actual_seq_lengths_kv", "actual_shared_prefix_len"],
        )
    })

if hasattr(torch.ops.npu, "npu_fused_infer_attention_score_v2"):
    _REPLACE_FUNC_MAP.update({torch.ops.npu.npu_fused_infer_attention_score_v2.default:
        StaticWorkspaceReplaceFunc(
            get_workspace=torch.ops.npu._npu_fused_infer_attention_score_v2_get_max_workspace.default,
            out_operator=torch.ops.npu.npu_fused_infer_attention_score_v2.out,
            workspace_keys=["workspace"],
            output_keys=["attention_out", "softmax_lse"],
            updated_param_keys=["actual_seq_qlen", "actual_seq_kvlen"],
        )
    })


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

    if all(idx[1] for idx in unupdated_input_index):
        # all the sym input are symbol
        def gen_unupdated_sym_input_key(*args: Any, **kwargs: Any):
            input_sym_list = []
            for idx in unupdated_input_index:
                input_sym_list.append(args[idx[0]])
            return str(input_sym_list)

        return gen_unupdated_sym_input_key

    # input is mix case for symbol and tensor
    def gen_unupdated_input_key(*args: Any, **kwargs: Any):
        input_shape_list = []
        for idx in unupdated_input_index:
            if idx[1]:
                # symbol input
                input_shape_list.append(args[idx[0]])
            else:
                # tensor input
                input_shape_list.append(list(args[idx[0]].shape))

        return str(input_shape_list)

    return gen_unupdated_input_key


def get_unupdated_input_fn(unupdated_sym_input_index, parameter_user_inputs, config):
    enable_parameter_frozen = "frozen_parameter" in config.keys() and config["frozen_parameter"] == "1"
    if enable_parameter_frozen or parameter_user_inputs is None or len(parameter_user_inputs) == 0:
        return gen_unupdate_input_func(unupdated_sym_input_index)
    else:
        # Normally, we can enable frozen_parameter to optimize over head time, so this branch is generally not used.
        unupdated_input_fn = gen_unupdate_input_func(unupdated_sym_input_index)

        def gen_input_key_with_parameter_addr(*args: Any, **kwargs: Any):
            input_key = unupdated_input_fn(*args, **kwargs)
            addr_list = []
            for idx in parameter_user_inputs:
                addr_list.append(args[idx].data_ptr())
            return input_key + str(hash(tuple(addr_list)))

        return gen_input_key_with_parameter_addr


def get_unupdated_sym_input_index(graph_module: torch.fx.GraphModule, all_sym_input_idx):
    updated_op_params = {}
    for func_iter in _REPLACE_FUNC_MAP.values():
        if len(func_iter.workspace_keys) > 0:
            updated_op_params[func_iter.get_workspace] = func_iter.updated_param_keys
        updated_op_params[func_iter.out_operator] = func_iter.updated_param_keys
    logger.debug("In graph[%s], all updated inputs user nodes and params: %s.", id(graph_module), updated_op_params)

    unupdated_sym_input_index = set()
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
        if 'val' not in node.meta:
            logger.debug('Find placeholder node with no val in meta, placeholder index=%s, value=%s, type=%s',
                         data_idx, node, type(node).__name__)
            continue
        node_meta = node.meta['val']
        if not have_sym_in_meta(node_meta):
            continue
        logger.debug("In graph[%s], the %s th meta input[%s] have sym, with all users[%s].",
                     id(graph_module), data_idx, node_meta, node.users)
        if len(node.users) == 0:
            continue

        the_unupdated_user = None
        for user_node in node.users:
            if user_node.target not in updated_op_params.keys():
                the_unupdated_user = user_node
                break
            unupdated_inputs = get_node_all_placeholder_inputs(user_node,
                                                               excluded_kwargs=updated_op_params[user_node.target])
            if node in unupdated_inputs:
                the_unupdated_user = user_node
                break
        if the_unupdated_user is None:
            continue

        logger.debug("In graph[%s], the %s th meta input[%s: %s] have unupdated user[%s].",
                     id(graph_module), data_idx, node.name, node_meta, the_unupdated_user.name)
        if is_sym(node_meta):
            unupdated_sym_input_index.add((data_idx, True))
            continue
        for dim in node_meta.size():
            if not is_sym(dim):
                continue
            idx = all_sym_input_idx.get(dim.node.expr, None)
            if idx is not None:
                unupdated_sym_input_index.add((idx, True))
            else:
                unupdated_sym_input_index.add((data_idx, False))

    unupdated_sym_input_index = list(unupdated_sym_input_index)
    unupdated_sym_input_index.sort()
    logger.debug("In graph[%s], all unupdated symbol input index is %s.",
                 id(graph_module), unupdated_sym_input_index)
    return unupdated_sym_input_index


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


class CapturedGraphUpdateAndReplay(nn.Module):
    """
    Module for replaying captured graphs with dynamic updates.
    """

    _update_stream: Optional["torch.npu.Stream"] = None

    def __init__(self, replay_graph: Any, updated_input_func: Callable, updated_node_infos: List):
        super().__init__()
        self._replay_graph = replay_graph
        self._updated_input_func = updated_input_func
        self._updated_node_infos = updated_node_infos
        if self.__class__._update_stream is None:
            self.__class__._update_stream = torch.npu.Stream(priority=-1)

    def forward(self, *args: Any, **kwargs: Any):
        self._replay_graph.replay()

        replay_stream_id = torch.npu.current_stream().stream_id
        while self.__class__._update_stream.stream_id == replay_stream_id:
            self.__class__._update_stream = torch.npu.Stream(priority=-1)
            logger.info(f"Update the stream for parameter, replay stream id: {replay_stream_id}, "
                        f"update stream id: {self.__class__._update_stream.stream_id}.")

        updated_kwargs = self._updated_input_func(*args, **kwargs)
        logger.debug("In AclGraph running, all updated op_name and param: %s.", updated_kwargs)
        if len(updated_kwargs) == 0:
            return

        with torch.npu.stream(self.__class__._update_stream):
            for node_info in self._updated_node_infos:
                torch.npu.graph_task_update_begin(self.__class__._update_stream, node_info.handle)
                node_kwargs = dict(node_info.kwargs)
                for key in node_info.updated_param_name:
                    node_kwargs[key] = updated_kwargs[node_info.node_name][key]
                node_info.updated_func(*node_info.args, **node_kwargs)
                torch.npu.graph_task_update_end(self.__class__._update_stream)
                self.__class__._update_stream.record_event(node_info.event)

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


# Record the fx nodes created based on symbolic expressions in each graph module
# data format: {graph id : {symbolic expressions : fx node}}
_GLOBAL_SYM_EXPR_2_NODE_MAP = {}


def construct_fx_node_shape(ori_shape: List, sym_inputs: dict, graph_id: int) -> List:
    global _GLOBAL_SYM_EXPR_2_NODE_MAP
    empty_shape = []
    for dim in ori_shape:
        if is_constant(dim):
            # only for real constant value
            empty_shape.append(dim)
        elif isinstance(dim.node.expr, sympy.Symbol) or str(dim.node.expr).isdigit():
            # for single symbol or number symbol, such as: s0, sym(5)
            if dim.node.expr not in sym_inputs.keys():
                raise RuntimeError(f'Unexpected sym output shape {dim} '
                                   f'which is not in all sym inputs dict {sym_inputs} of graph[{graph_id}].')
            empty_shape.append(sym_inputs[dim.node.expr])
        else:
            # only for sym expr, such as: 32*(s0//32)
            if graph_id not in _GLOBAL_SYM_EXPR_2_NODE_MAP.keys():
                _GLOBAL_SYM_EXPR_2_NODE_MAP[graph_id] = {}  # init node map of current graph graph id

            if str(dim) in _GLOBAL_SYM_EXPR_2_NODE_MAP[graph_id].keys():
                empty_shape.append(_GLOBAL_SYM_EXPR_2_NODE_MAP[graph_id][str(dim)])
                continue
            sym_set = dim.node.expr.free_symbols
            proxy_nodes = {}
            for sym_i in sym_set:
                globals()[str(sym_i)] = Proxy(sym_inputs[sym_i])
            expr_proxy = eval(str(dim))
            empty_shape.append(expr_proxy.node)
            _GLOBAL_SYM_EXPR_2_NODE_MAP[graph_id][str(dim)] = expr_proxy.node
            logger.debug("Record all created sym expr and fx node %s in graph[%s].",
                         _GLOBAL_SYM_EXPR_2_NODE_MAP[graph_id], graph_id)

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
        empty_shape = construct_fx_node_shape(meta_outputs[i].shape, sym_inputs, id(graph_module))
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

        if not hasattr(node, "meta") or 'val' not in node.meta:
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
    from torchair._acl_concrete_graph.graph_pass import inplaceable_npu_ops
    inplace_ops_mutated_indices = {inplaceable_op.inplace_op: inplaceable_op.mutated_arg
                                   for _, inplaceable_op in inplaceable_npu_ops.items()}

    inplace_node_args_list = []
    placeholder_args = set()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholder_args.add(node.target)
        elif _is_inplace_node(node):
            inplace_node_args_list.append(node.args[0].name)
        elif node.target in inplace_ops_mutated_indices:
            for idx in inplace_ops_mutated_indices[node.target]:
                inplace_node_args_list.append(node.args[idx].name)
        # TO DO 1: add process for inplace op generated by auto_functionalized_v2
        # TO DO 2: add process for case: inplace op input is a view of input

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
    # To ensure that npu_stream_switch obtains the correct stream in a cache_compile scenario
    user_stream_label: Set[str] = field(default_factory=set)
    # dict for inputs and outputs which are same tensor.
    userinput_ref_with_output: Dict[int, List] = field(default_factory=dict)

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
        self.userinput_ref_with_output_storages_ptr = set()

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
        self._user_stream_label = None
        self._input_base_format = None
        self._userinput_ref_with_output: Dict[int, List] = {}

    def __call__(self, *args, **kwargs):
        # get graph_key and capture
        fn_key = self.compile(*args, **kwargs)

        # fall back to eager when static_capture_size_limit is exceeded
        if self.fallback_to_eager:
            with record_function("fx_run_eagerly"):
                return self.fx_run_eagerly(*args, **kwargs)

        # input process
        with record_function("process_input"):
            self.process_input(fn_key, *args)

        # run/replay
        with record_function("acl_graph_replay"):
            self.run(fn_key, *args, **kwargs)

        return self.reconstruct_outputs(fn_key)


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
        return self.fx_forward(*args, **kwargs)

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

        '''
        NB: Do not create additional capture streams arbitrarily
        If the user does not explicitly specify the stream to be used for capture, all graphs use the default stream.
        The goal is to organize the memory of all captured graphs on the same stream. Based on the above premise,
        when users specify the same memory pool in multiple graphs,all graphs can reuse memory.
        Otherwise, even within the same memory pool, memory reuse may fail
        because those memory blocks are in different streams.
        '''
        self._stream = aclgraph_cache_info.stream
        self._capture_error_mode = aclgraph_cache_info.capture_error_mode
        self._num_warmup_iters = aclgraph_cache_info.num_warmup_iters
        self._fx_graph_name = aclgraph_cache_info.fx_graph_name
        self._tagged_event_names = aclgraph_cache_info.tagged_event_names
        self._user_stream_label = aclgraph_cache_info.user_stream_label
        self._userinput_ref_with_output = aclgraph_cache_info.userinput_ref_with_output

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
                compile_static_kernel(self.fx_forward, *args, build_dir=path, **kwargs)

            self._unupdated_input_func = get_unupdated_input_fn(self._unupdated_sym_input_index, self._parameter_user_inputs, self.config)
            self._updated_input_func = get_updated_ops_fn(self._ops_update_rulers)
            self._captured = True

            # In the current version, the initialization of mem pool requires an explicit call to capture.
            # In versions greater than 2.6, the initialization can be completed directly when creating the mem pool.
            import torch_npu
            g = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(g, pool=self.pool, stream=self.stream):
                pass
            # record the original memory state before capture,
            # and it will be used to restore the mem state when capturing another acl graph for different shape.
            self._original_mem_state = torch_npu._C._npu_getCheckpointState(self.device, self.pool)

        if self.fallback_to_eager:
            # when falling back to eager, no need to calculate graph key
            return "fallback_to_eager"

        saved_acl_graph = None
        # get graph key based on unupdated sym input shape or value
        graph_key = self._unupdated_input_func(*args, **kwargs)
        if graph_key in self._graphs_meta.keys():
            logger.info('Find captured AclGraph{id: %s} of fx_graph %s with graph key {%s}.',
                        id(self.graph[graph_key]), self.name, graph_key)
            if self.is_need_to_recapture(graph_key, *args):
                logger.debug('The current AclGraph needs to be recaptured for fx_graph %s with graph key {%s}.',
                             self.name, graph_key)
                # save graph_meta, release resources after recapture
                saved_acl_graph = self._graphs_meta[graph_key].acl_graph
            else:
                logger.debug('The current AclGraph no needs to be recaptured for fx_graph %s with graph key {%s}.',
                             self.name, graph_key)
                return graph_key

        # handle retained outputs from last capture when only one graph key and mempool reuse enabled
        enable_mempool_reuse = not ("disable_mempool_reuse_in_same_fx" in self.config.keys() and self.config[
            "disable_mempool_reuse_in_same_fx"] == "1")
        if enable_mempool_reuse and len(self._graphs_meta) == 1:
            (last_graph_key,) = self._graphs_meta.keys()
            if graph_key != last_graph_key:
                # reset output weakref from last capture to retained outputs
                for idx, retained_output in enumerate(self._graphs_meta[last_graph_key].retained_outputs):
                    output_ref = self._graphs_meta[last_graph_key].outputs_weakref[idx]()
                    if output_ref is None or isinstance(output_ref, torch.Tensor):
                        self._graphs_meta[last_graph_key].outputs_weakref[idx].swap_weakref(retained_output)
            self._graphs_meta[last_graph_key].retained_outputs = None

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
        if saved_acl_graph is not None:
            saved_acl_graph.reset()

        return graph_key

    def reset_captured_graph(self):
        # Do sync before we reset all captured aclgraphs, just like what was done before capture
        torch.npu.synchronize()
        logger.info('Current fx_graph %s memory pool is %s. Before reset, the current memory state is {%s}.',
                    self.name, self.pool, LazyMessage(debug_mem_state))

        for _, graph_meta in self._graphs_meta.items():
            graph_meta.acl_graph.reset()
        self._graphs_meta = {}
        gc.collect()
        torch.npu.empty_cache()

        logger.info('Current fx_graph %s memory pool is %s. After reset, the current memory state is {%s}.',
                    self.name, self.pool, LazyMessage(debug_mem_state))

    def get_stale_list_from_weakref(self):
        # get alive stale outputs depending on num of graph key
        stale_storage_set = set()
        for key, _ in self._graphs_meta.items():
            if self._graphs_meta[key].outputs_weakref is None:
                continue
            for output_ref in self._graphs_meta[key].outputs_weakref:
                ref = output_ref()
                if ref is not None and isinstance(ref, torch.Tensor) and \
                        ref.untyped_storage()._cdata in self.stale_storages_ptr:
                    stale_storage_set.add(ref.untyped_storage()._cdata)
        return list(stale_storage_set)

    def compile_for_graph_key(self, graph_key, *args: Any, **kwargs: Any):
        """
        Compile the FX graph into another new ACL graph instance with specific graph key.

        Args:
            graph_key (str): Unique identifier for the captured ACL graph.
            *args: Input arguments for FX graph.
            **kwargs: Keyword arguments for FX graph.
        """

        # get stale storages from last run for current fx graph
        stale_storages = self.get_stale_list_from_weakref()

        import torch_npu
        self._graphs_meta[graph_key] = GraphMeta(graph_key=graph_key,
                                                 acl_graph=torch_npu.npu.NPUGraph(),
                                                 replay_func=None,
                                                 userinputs_meta={},
                                                 userinputs_metatensor={},
                                                 userinputs_weakref={},
                                                 outputs_meta=[],
                                                 outputs_weakref=[],
                                                 mem_state_after_capture=None,
                                                 captured_parameter={},
                                                 captured_mutated_inputs={},
                                                 captured_userinput_ref_with_output={},
                                                 )
        # record the parameter address
        for parameter_idx in self._parameter_user_inputs:
            self.graphs_meta[graph_key].captured_parameter.setdefault(parameter_idx, args[parameter_idx].data_ptr())

        for input_name, input_idx in self._user_inputs_mapping.items():
            # record the mutated_inputs address for subsequent comparison to determine if recapture is necessary
            if input_name in self._mutated_user_inputs:
                self.graphs_meta[graph_key].captured_mutated_inputs.setdefault(input_idx, args[input_idx].data_ptr())
        
        for ref_idx in self._userinput_ref_with_output.keys():
            self.graphs_meta[graph_key].captured_userinput_ref_with_output.setdefault(ref_idx, args[ref_idx])
            self.userinput_ref_with_output_storages_ptr.add(args[ref_idx].untyped_storage()._cdata)
        self.graphs_meta[graph_key].captured_output_idx_ref_with_userinput = set(itertools.chain(*self._userinput_ref_with_output.values()))

        logger.debug('No find captured AclGraph{id: %s} of fx_graph %s with graph key {%s}, and start to capture it.',
                     id(self.graph[graph_key]), self.name, graph_key)

        # set to common original memory state before capture
        # only enable mempool reuse when graph key exceeds 1
        enable_mempool_reuse = not ("disable_mempool_reuse_in_same_fx" in self.config.keys() and self.config[
            "disable_mempool_reuse_in_same_fx"] == "1")
        if enable_mempool_reuse and len(self._graphs_meta) > 1:
            logger.debug('Start setting to original memory state for fx_graph %s with graph key{%s}. '
                         'The stale storage is %s, and the current memory state is {%s}.',
                         self.name, graph_key, stale_storages, LazyMessage(debug_mem_state))
            torch_npu._C._npu_setCheckpointPoolState(self.device, self._original_mem_state, stale_storages, [])
            logger.debug('After setting to original memory state for fx_graph %s with graph key{%s}. '
                         'The current memory state is {%s}.',
                         self.name, graph_key, LazyMessage(debug_mem_state))
            for ptr in stale_storages:
                self.stale_storages_ptr.discard(ptr)

        # start capture aclgraph
        with record_function("acl_graph_capture"):
            captured_outputs = self.capture(graph_key, *args, **kwargs)

        # The captured output and input tensors will not be held indefinitely
        # and they will be terminated after this capture.
        self._graphs_meta[graph_key].mem_state_after_capture = torch_npu._C._npu_getCheckpointState(self.device,
                                                                                                    self.pool)
        logger.debug('After capturing fx_graph %s for graph key{%s} to AclGraph{id: %s}, '
                     'memory pool reuse is %s, the memory state is {%s}.',
                     self.name, graph_key, id(self.graph[graph_key]),
                     "enable" if enable_mempool_reuse else "disable", LazyMessage(debug_mem_state))
        if self.config.get('clone_input', "1") == "1":
            # Note that userinputs are not kept alive; they are reconstructed for process_input if clone_input is True.
            self._graphs_meta[graph_key].retained_userinputs.clear()
        if enable_mempool_reuse and len(self._graphs_meta) > 1:
            del captured_outputs
        else:
            # retain graph output when only one graph key or mempool reuse disabled
            self._graphs_meta[graph_key].retained_outputs = captured_outputs

    def capture(self, graph_key, *args: Any, **kwargs: Any):
        """
        Captures the execution of the FX graph into an ACL graph instance.

        Args:
            graph_key (str): Unique identifier for the captured graph.
            *args: Input arguments for graph capture.
            **kwargs: Keyword arguments for graph capture.
        """
        args_list = list(args)
        import torch_npu
        # Clear _updated_node_infos(list of UpdatedNodeInfo objects) before capture.
        # Each object stores graph task group handles/events for the current capture,
        # and must be updated on recapture.
        self._updated_node_infos.clear()
        with torch_npu.npu.graph(self.graph[graph_key], pool=self.pool, stream=self.stream,
                                 capture_error_mode=self.capture_error_mode):
            for input_name, input_idx in self._user_inputs_mapping.items():
                if input_name in self._mutated_user_inputs:
                    continue
                # if input is an alias of output, its metatensor will not be create.(it will be retained).
                if input_idx in self._userinput_ref_with_output.keys():
                    continue
                if isinstance(args_list[input_idx], torch.Tensor):
                    weak_ref = WeakRef(None)
                    # If clone_input is set to True, each user input will have a unique data pointer, preventing
                    # sharing between inputs. For capture, the original data_ptr in args_list[input_idx] is swapped
                    # with a new one generated by torch.empty_like. This ensures the aclgraph records the new data
                    # pointer for later reuse. Finally, the weak reference to each user input is set to None,
                    # as the inputs will be reconstructed during processing.
                    # If clone_input is set to False, records the original data_ptr in args_list[input_idx].
                    if self.config.get('clone_input', "1") == "1" and args_list[input_idx].is_npu:
                        args_list[input_idx] = torch.empty_like(args_list[input_idx])
                    # Uses retained_userinputs to ensure capture of the args_list.
                    self._graphs_meta[graph_key].retained_userinputs.setdefault(input_idx, args_list[input_idx])
                else:
                    weak_ref = WeakRef(args_list[input_idx])
                self._graphs_meta[graph_key].userinputs_weakref.setdefault(input_idx, weak_ref)
                self._graphs_meta[graph_key].userinputs_meta.setdefault(input_idx,
                                                                        get_tensor_metadata(args_list[input_idx]))
                self._graphs_meta[graph_key].userinputs_metatensor.setdefault(input_idx,
                    reconstruct_from_tensor_metadata(self._graphs_meta[graph_key].userinputs_meta[input_idx]))
            captured_outputs = self.fx_forward(*args_list, node_info=self._updated_node_infos, is_capturing=True,
                                               **kwargs)
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

        logger.debug('AclGraph{id: %s} of fx_graph %s with graph key {%s}, has {%s} parameters and {%s} mutated_inputs,'
                     ' and {%s} input alias, '
                     'all the input meta info are %s.',
                     id(self.graph[graph_key]), self.name, graph_key,
                     len(self.graphs_meta[graph_key].captured_parameter.keys()),
                     len(self.graphs_meta[graph_key].captured_mutated_inputs.keys()),
                     len(self.graphs_meta[graph_key].captured_userinput_ref_with_output.keys()),
                     self._graphs_meta[graph_key].userinputs_meta)
        logger.info('Success to capture fx_graph %s for graph key{%s}. '
                    'Start to run AclGraph{id: %s} with the updated node num {%s}.',
                    self.name, graph_key, id(self.graph[graph_key]), len(self._updated_node_infos))

        if os.getenv("TORCH_COMPILE_DEBUG", "0") == "1":
            try:
                dir_path = DebugContext.current_path()
                os.makedirs(dir_path, exist_ok=True)
                timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S%f")
                file_name = f"{self.name}_id_{id(self.graph[graph_key])}_rank_{self._device}_pid_{os.getpid()}_ts_{timestamp}.json"
                self.graph[graph_key].debug_dump(os.path.join(dir_path, file_name))
            except Exception:
                logger.exception("Aclgraph for fx_graph %s json debug dump failed", self.name)

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
        # reconstruct inputs
        # Does it has to enable_mempool_reuse when reuse inputs? - No
        # 1. If the memory pool is not user-specified (i.e., one pool per FX graph), in scenarios with multiple ACL graphs,
        # input memory might occupy the output memory of other ACL graphs. In this case, if input_retainedis required
        # to prevent conflicts with outputs captured by other ACL graphs, it defeats the purpose of memory reuse.
        # It would be simpler to use the original user inputs directly. Therefore, there is no need to call
        # set_to_original_state_before_reconstruct(graph_key)to mark alive outputs as stale, since the inputs and
        # outputs of the sameACL graph will not occupy the same memory block.
        # 2. If the memory pool is user-specified and output memory reuse is disabled, output memory is retained.
        # This ensures that input memory allocation does not claim memory blocks occupied by outputs (from the same
        # or other graphs). Thus, reconstructing input memory will not corrupt the outputs of the current or
        # other graphs.
        with timer(f"{self.name} process inputs reconstruct inputs"):
            if self.config.get('clone_input', "1") == "1":
                # When a memory pool is shared among aclgraph instances (from the same or different FX graphs),
                # resetting the memory state before input reconstruction is unnecessary. Safety is guaranteed
                # by the runtime constraint that only one aclgraph executes at a time. Since inputs are
                # assigned before each replay and destroyed afterwards, there is no risk of state conflict.
                reconstructed_inputs = self._graphs_meta[graph_key].userinputs_metatensor
            else:
                # Use retained_userinputs and userinput_ref_with_output if clone_input is set to False.
                reconstructed_inputs = self._graphs_meta[graph_key].retained_userinputs
            reconstructed_inputs.update(self._graphs_meta[graph_key].captured_userinput_ref_with_output or {})
                 
        # foreach copy
        dst_tensors = []
        src_tensors = []
        with timer(f"{self.name} process inputs foreach copy"):
            for input_name, input_idx in self._user_inputs_mapping.items():
                if input_name in self._mutated_user_inputs:
                    continue
                capture_input = reconstructed_inputs[input_idx]
                replay_arg = args[input_idx]
                if capture_input.data_ptr() != replay_arg.data_ptr():
                    dst_tensors.append(capture_input)
                    src_tensors.append(replay_arg)
            if len(dst_tensors) > 1:
                if self._input_base_format is None:
                    if is_inputs_base_format(dst_tensors) and is_inputs_base_format(src_tensors):
                        self._input_base_format = True
                    else:
                        self._input_base_format = False
                if self._input_base_format:
                    torch._foreach_copy_(dst_tensors, src_tensors, non_blocking=True)
                else:
                    for i, dst_tensors in enumerate(dst_tensors):
                        dst_tensors.copy_(src_tensors[i])

            elif len(dst_tensors) == 1:
                dst_tensors[0].copy_(src_tensors[0])


    def run(self, graph_key, *args, **kwargs):
        self._graphs_meta[graph_key].replay_func(*args, **kwargs)

    def is_need_to_recapture(self, graph_key, *args: Any):
        # There is no need to recapture the ACLGraph when the data addresses of user inputs
        # (which are aliases to outputs) change.
        # When the memory addresses of mutated_inputs and parameter type inputs change
        # recapture the aclgraph to reduce copy time and improve performance
        for idx, mutated_ptr in self._graphs_meta[graph_key].captured_mutated_inputs.items():
            if mutated_ptr != args[idx].data_ptr():
                return True
        return False

    def set_stale_storages_ptr_exclude_ref_userinputs(self, retained_outputs: List, graph_key: str):
        self.stale_storages_ptr = set()
        output_idx_ref_with_userinput = self._graphs_meta[graph_key].captured_output_idx_ref_with_userinput
        for idx, retained_output in enumerate(retained_outputs):
            if idx not in output_idx_ref_with_userinput and \
                    isinstance(retained_output, torch.Tensor):
                self.stale_storages_ptr.add(retained_output.untyped_storage()._cdata)

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
            self.set_stale_storages_ptr_exclude_ref_userinputs(self._graphs_meta[graph_key].retained_outputs, graph_key)
            return self._graphs_meta[graph_key].retained_outputs

        if len(self.graphs_meta) == 1:
            # no need to reconstruct output when only one graph key
            logger.debug('When mempool reuse is enabled in fx_graph %s for graph key{%s} '
                         'and there is only one graph meta captured, all the outputs are retained.',
                         self.name, graph_key)
            self.set_stale_storages_ptr_exclude_ref_userinputs(self._graphs_meta[graph_key].retained_outputs, graph_key)
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
            self.set_stale_storages_ptr_exclude_ref_userinputs(ret, graph_key)
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
            outputs = [out.clone() if (isinstance(out, torch.Tensor) and \
                out.untyped_storage()._cdata not in self.userinput_ref_with_output_storages_ptr)\
                    else out for out in outputs]
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
                tensor_i.set_(alive_outputs[ptr].untyped_storage(),
                              tensor_i.storage_offset(), tensor_i.shape, tensor_i.stride())
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
        # Graph inputs should be freed immediately after replay, thus there will be no alive inputs.
        # Graph outputs may be freed by user, we just get tensor that still alive after last execution.
        stale_storage_set = set()
        for key, _ in self._graphs_meta.items():
            idx_ref_with_userinput = self._graphs_meta[key].captured_output_idx_ref_with_userinput
            for idx, output_ref in enumerate(self._graphs_meta[key].outputs_weakref):
                ref = output_ref()
                if ref is not None and isinstance(ref, torch.Tensor) and \
                        ref.untyped_storage()._cdata in self.stale_storages_ptr and \
                        idx not in idx_ref_with_userinput:
                    stale_storage_set.add(ref.untyped_storage()._cdata)
        other_graph_stale_storages = list(stale_storage_set)

        import torch_npu
        if len(other_graph_stale_storages) > 0:
            logger.debug('Before reset fx_graph %s outputs stale storages cdata %s to original memory state '
                         'for AclGraph{id: %s} with the current graph key{%s}, and the current memory state is {%s}.',
                         self.name, other_graph_stale_storages, id(self.graph[graph_key]), graph_key,
                         LazyMessage(debug_mem_state))
            # Reset other graph live tensors to stale storages
            torch_npu._C._npu_setCheckpointPoolState(self.device, self._original_mem_state,
                                                     other_graph_stale_storages, [])
            self.stale_storages_ptr = set()

        logger.debug('After reset fx_graph %s outputs stale storages, '
                     'for AclGraph{id: %s} with the current graph key{%s}, and the current memory state is {%s}.',
                     self.name, id(self.graph[graph_key]), graph_key, LazyMessage(debug_mem_state))

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
            if isinstance(output_i, torch.Tensor) and \
                output_i.untyped_storage()._cdata not in self.userinput_ref_with_output_storages_ptr:
                if output_i.untyped_storage().data_ptr() not in all_reconstructed_storages_ptr.keys():
                    reconstructed_outputs_to_add_deleter.append(output_i.untyped_storage()._cdata)
                    all_reconstructed_storages_ptr[output_i.untyped_storage().data_ptr()] = [output_i]
                else:
                    all_reconstructed_storages_ptr[output_i.untyped_storage().data_ptr()].append(output_i)
        # add reconstruct inputs to deleter
        if self.config.get('clone_input', "1") == "1":
            reconstructed_inputs = self.reconstruct_inputs(graph_key)
            for idx, input_i in reconstructed_inputs.items():
                if isinstance(input_i, torch.Tensor):
                    reconstructed_outputs_to_add_deleter.append(input_i.untyped_storage()._cdata)

        # Currently we deallocate on instead of allowing stale recordings
        stale_storages: List[int] = []
        import torch_npu
        logger.debug('Before reconstructing fx_graph %s outputs for graph key{%s}, '
                     'the storages to add deleter are %s, the memory state is {%s}.',
                     self.name, graph_key, reconstructed_outputs_to_add_deleter, LazyMessage(debug_mem_state))
        # Set reconstructed outputs deleter fn
        torch_npu._C._npu_setCheckpointPoolState(self.device, self._graphs_meta[graph_key].mem_state_after_capture,
                                                 stale_storages, reconstructed_outputs_to_add_deleter)

        # When multiple Python tensors have the same storage ptr, they should have the same storage object
        for _, tensor_list in all_reconstructed_storages_ptr.items():
            if len(tensor_list) < 2:
                continue
            for tensor_i in tensor_list[1:]:
                tensor_i.set_(tensor_list[0].untyped_storage(),
                              tensor_i.storage_offset(), tensor_i.shape, tensor_i.stride())

        self.stale_storages_ptr.update(reconstructed_outputs_to_add_deleter)
        logger.debug('After reconstructing fx_graph %s outputs for graph key{%s}, '
                     'the memory state is {%s}.',
                     self.name, graph_key, LazyMessage(debug_mem_state))

    def reconstruct_inputs(self, graph_key):
        reconstructed_inputs = {}
        # reconstruct input tensor based on tensor meta info.
        for idx, input_meta in self._graphs_meta[graph_key].userinputs_meta.items():
            input_ref = self._graphs_meta[graph_key].userinputs_weakref[idx]()
            if input_ref is None or isinstance(input_ref, torch.Tensor):
                input_i = reconstruct_from_tensor_metadata(input_meta)
                # weakref of input_i will be None once replay is finished.
                # No inputs will be alive cause no one will keep them.
                self._graphs_meta[graph_key].userinputs_weakref[idx].swap_weakref(input_i)
                reconstructed_inputs[idx] = input_i
            else:
                # other non tensor obj can be returned directly.
                reconstructed_inputs[idx] = input_ref
        return reconstructed_inputs