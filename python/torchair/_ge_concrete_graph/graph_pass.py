import copy
import logging
from collections import namedtuple, defaultdict
from typing import Any, Dict, List, Tuple, Union, Callable, Set
import torch
from torchair.core.utils import logger
from torchair._ge_concrete_graph.ge_ir_pb2 import GraphDef, TensorDescriptor, TensorDef, OpDef
from torchair.ge._ge_graph import _ge_proto_dtype_to_ge_dtype, compat_as_bytes, torch_type_to_ge_type, \
    _SymPackInput, _ValueType, _GeInputInfo, GeGraph, ControlTensor, _TensorInput
from torchair.ge._ge_graph import Tensor as GeTensor
from torchair._ge_concrete_graph.utils import Placement, update_op_input_name_from_mapping, generate_shape_from_tensor

from . import ge_apis as ge


def _find_pack_and_data_ops(graph: GraphDef):
    host_data_ops: Dict[str, int] = {}
    sym_pack_ops: List[OpDef] = []
    for op in graph.op:
        if op.type == "Data" and op.input_desc[0].device_type == "CPU":
            host_data_ops[GeTensor(op).tensor] = op.attr["index"].i
        elif op.type == "Pack" and op.attr['_inputs_all_sym'].b is True:
            if op.attr["axis"].i != 0:
                raise AssertionError(f"sym pack op attr axis must be 0.")
            sym_pack_ops.append(op)
    logger.info(f"find all host data ops {host_data_ops.keys()}, and sym pack ops {[op.name for op in sym_pack_ops]}.")

    return host_data_ops, sym_pack_ops


def _find_data_of_same_pack(datas: List[Tuple[GeTensor, OpDef]], pack_op: OpDef):
    for data_tensor, recorded_pack in datas:
        if len(recorded_pack.input) != len(pack_op.input):
            continue
        if all([recorded_pack.input[i] == pack_op.input[i] for i in range(len(recorded_pack.input))]):
            return data_tensor
    return None


def optimize_sym_pack(graph: GraphDef):
    host_data_ops, sym_pack_ops = _find_pack_and_data_ops(graph)
    additional_data: List[Tuple[GeTensor, OpDef]] = []
    pack_to_data_name_mapping = {}
    for pack_op in sym_pack_ops:
        new_data = _find_data_of_same_pack(additional_data, pack_op)
        if new_data is None:
            if not all([pack_in in host_data_ops for pack_in in pack_op.input]):
                raise AssertionError("Find undefined sym pack op which inputs are not data ops.")
            pack_input_idx_list = [host_data_ops[pack_in] for pack_in in pack_op.input]

            with graph:
                new_data = ge.Data(index=graph.num_inputs,
                                   dtype=_ge_proto_dtype_to_ge_dtype(pack_op.input_desc[0].dtype),
                                   shape=[len(pack_op.input)],
                                   placement='CPU',
                                   node_name=None)
                input_info = _GeInputInfo(value_type=_ValueType.TENSOR, func=_SymPackInput(pack_input_idx_list),
                                          shape=[len(pack_op.input)], device_type="CPU")
                graph.record_input_info(new_data.node.name, input_info)
            if GeTensor(pack_op).meta is not None:
                new_data.set_meta(GeTensor(pack_op).meta)
            additional_data.append((new_data, pack_op))
            logger.debug("create new data op %s with input list [%s] "
                         "to replace pack op %s.", new_data.node.name, pack_input_idx_list, pack_op.name)
        else:
            logger.debug("skip create data, set data op %s to replace pack op %s.", new_data.node.name, pack_op.name)

        pack_to_data_name_mapping[pack_op.name] = new_data.node.name
        graph.op.remove(pack_op)
    update_op_input_name_from_mapping(graph, pack_to_data_name_mapping)


ref_op_info = namedtuple("ref_op_info", ["op_def", "output_id", "ref_input_name", "output_ref_count", "is_net_output"])


def _get_output_to_input_ref_idx(op: OpDef) -> Dict[int, int]:
    """
    If there are more inplace op that need to be optimized similarly, it is necessary to
    add a mapping relationship between output and references input in this function.
    """
    ref_idx_mapping: Dict[int, int] = {}
    if op.type == "Scatter" or op.type == "QuantUpdateScatter" or op.type == "ScatterNdUpdate":
        ref_idx_mapping[0] = 0
    elif op.type == "ScatterList":
        for i in range(len(op.output_desc)):
            ref_idx_mapping[i] = i

    return ref_idx_mapping


def _find_ref_ops_and_io_ops(graph: GraphDef):
    data_output_count: Dict[str, Tuple[int, int]] = {}
    tensormove_ops: Dict[str, OpDef] = {}
    assign_ops: Dict[str, OpDef] = {}
    ref_ops: Dict[str, Tuple[OpDef, int, str]] = {}
    ref_output_count: Dict[str, int] = {}
    net_output: OpDef = None
    for op in graph.op:
        if op.type == "Data":
            data_output_count[GeTensor(op).tensor] = (op.attr["index"].i, 0)
        elif op.type == "TensorMove":
            tensormove_ops[GeTensor(op).tensor] = op
        elif op.type == "Assign":
            assign_ops[GeTensor(op).tensor] = op
        elif op.type == "NetOutput":
            net_output = op
        else:
            ref_io_mapping = _get_output_to_input_ref_idx(op)
            for output_id, input_id in ref_io_mapping.items():
                ref_ops[GeTensor(op, output_id).tensor] = (op, output_id, op.input[input_id])
                ref_output_count[GeTensor(op, output_id).tensor] = 0
    logger.info(f"Find all TensorMove ops: {tensormove_ops.keys()}, reference ops: {ref_ops.keys()} "
                f"and Assign ops: {assign_ops.keys()}.")

    for op in graph.op:
        if op.type == "Data":
            continue
        for _, op_in in enumerate(op.input):
            # Cmo OP prefetch address, Data output count ignore Cmo.
            if (op_in in data_output_count.keys()) and (op.type != "Cmo"):
                data_output_count[op_in] = (data_output_count[op_in][0], data_output_count[op_in][1] + 1)
            if op_in in ref_ops.keys():
                ref_output_count[op_in] = ref_output_count[op_in] + 1

    ref_op_infos: Dict = {}
    for op_tenosr in ref_ops.keys():
        ref_op_infos[op_tenosr] = ref_op_info(op_def=ref_ops[op_tenosr][0],
                                              output_id=ref_ops[op_tenosr][1],
                                              ref_input_name=ref_ops[op_tenosr][2],
                                              output_ref_count=ref_output_count[op_tenosr],
                                              is_net_output=True if op_tenosr in net_output.input else False)

    return data_output_count, tensormove_ops, assign_ops, ref_op_infos


'''
* ******************************************************************************************************
*         input                        Data --------|                  Data       
*           |                            |          |                    |        
*           |                        TensorMove     |                    |        
*           |                            |          |                    |        
*     Scatter_update_    --->      Scatter(ref op) /      --->     Scatter(ref op) 
*           |                         |      \    /                      |         
*       other_op                  other_op    \  /                   other_op      
*           |                         |      assign                      |         
*        output                   NET_OUTPUT                         NET_OUTPUT    
* ******************************************************************************************************
*         input                     Data --------|              Data --------|              Data       
*           |                         |          |                |          |                |        
*           |                     TensorMove     |                |          |                |        
*           |                         |          |                |          |                |        
*     Scatter_update_   --->    Scatter(ref op) /    --->   Scatter(ref op) /    --->    Scatter(ref op) 
*           |                      |      \    /               |      \    /                  |         
*           |                      |       \  /                |       \  /                   |      
*           |                      |      assign               |      assign                  |         
*        output                NET_OUTPUT                   NET_OUTPUT                    NET_OUTPUT    
* ******************************************************************************************************

Due to the limitation that the inplace operator cannot directly enter the fx graph, 
it will be converted into a non inplace operator and inplace copy for implementation.
However, this process will introduce two redundant data copies, so we need to optimize the graph
as shown in the above figure, to simplify the memory copy in the graph.
'''


def optimize_reference_op_redundant_copy(graph: GraphDef):
    data_ops, tensormove_ops, assign_ops, ref_op_infos = _find_ref_ops_and_io_ops(graph)
    ref_data_idx = []
    for assign_tensor, assign_op in assign_ops.items():
        if (assign_op.input[0] in data_ops.keys()):
            ref_data_idx.append(data_ops[assign_op.input[0]][0])
        if (assign_op.input[0] not in data_ops.keys()) or (assign_op.input[1] not in ref_op_infos.keys()):
            logger.debug("Assign op: %s is not used to update ref_op output to data op.", assign_tensor)
            continue
        ref_op, ref_idx, ref_op_input = ref_op_infos[assign_op.input[1]].op_def, \
                                        ref_op_infos[assign_op.input[1]].output_id, \
                                        ref_op_infos[assign_op.input[1]].ref_input_name
        if ref_op_input not in tensormove_ops.keys():
            logger.debug("ref_op input: %s type is not TensorMove, skip TensorMove optimization.", ref_op_input)
            continue
        tensormove_op = tensormove_ops[ref_op_input]
        if tensormove_op.input[0] != assign_op.input[0]:
            logger.debug("TensorMove input: %s is not Data to be assigned, skip optimization.", tensormove_op.input[0])
            continue
        data_output_count = data_ops[assign_op.input[0]][1]
        if (data_output_count != 2):
            logger.debug("Data op: %s output count %s is not equal to two, "
                         "which means the current Data op cannot be modified inplace, skip TensorMove optimization.",
                         assign_op.input[0], data_output_count)
            continue

        ref_out_count, is_net_output = ref_op_infos[assign_op.input[1]].output_ref_count, \
                                       ref_op_infos[assign_op.input[1]].is_net_output
        if is_net_output or ref_out_count == 1:
            logger.debug("Assign op: %s is used to copy %s to %s, update ref_op ref_input_%s from %s to %s.",
                         assign_tensor, assign_op.input[1], assign_op.input[0], ref_idx, ref_op_input,
                         tensormove_op.input[0])
            logger.debug("Ref_op ref_output_%s is a net_output, Assign op can not be removed, "
                         "only remove TensorMove op: %s.", ref_idx, ref_op_input)
            ref_op.input[ref_idx] = tensormove_op.input[0]
            graph.op.remove(tensormove_op)
        else:
            logger.debug("Assign op: %s is used to copy %s to %s, update ref_op ref_input_%s from %s to %s, "
                         "and remove Assign op: %s and TensorMove op: %s.",
                         assign_tensor, assign_op.input[1], assign_op.input[0], ref_idx, ref_op_input,
                         tensormove_op.input[0], assign_tensor, ref_op_input)
            ref_op.input[ref_idx] = tensormove_op.input[0]
            graph.op.remove(assign_op)
            graph.op.remove(tensormove_op)

    return ref_data_idx


_GLOBAL_NAME_MAP = defaultdict()


def _is_unique_input_name_in_graph(graph_name_key, input_node_name):
    if graph_name_key not in _GLOBAL_NAME_MAP:
        _GLOBAL_NAME_MAP.setdefault(graph_name_key, []).append(input_node_name)
        return True

    if input_node_name not in _GLOBAL_NAME_MAP[graph_name_key]:
        _GLOBAL_NAME_MAP[graph_name_key].append(input_node_name)
        return True

    return False


def replace_data_to_refdata(graph, ref_input_idx, inputs):
    replaced_map = {}
    for op in graph.op:
        if op.type != "Data":
            continue
        data_idx = op.attr["index"].i
        if data_idx in ref_input_idx:
            logger.debug("Try to deal with ref input %s:%s", op.name, data_idx)
            input_tensor = inputs[data_idx]
            shape_str = "_".join(str(x) for x in input_tensor.shape)
            stride_str = "_".join(str(x) for x in input_tensor.stride())
            offset_str = str(input_tensor.storage_offset())
            new_refdata_name = "RefData_" + shape_str + "_" + stride_str + "_" + offset_str + "_" + str(
                id(input_tensor))
            if _is_unique_input_name_in_graph(graph.name, new_refdata_name):
                logger.debug("Replace %s:RefData with %s:%s in graph %s", new_refdata_name, op.name, op.type,
                             graph.name)
                op.attr["_origin_data_name"].s = compat_as_bytes(op.name)
                replaced_map[op.name] = new_refdata_name
                op.name = new_refdata_name
                op.type = f"RefData"
            else:
                logger.warning(
                    f"Find repeated input {op.name} in graph {graph.name}, repeat name is {new_refdata_name}")

    update_op_input_name_from_mapping(graph, replaced_map)


def get_frozen_flag(input_infos):
    frozen_flag_list = []
    for idx, input_info in enumerate(input_infos):
        if (input_info.value_type == _ValueType.PARAMETER) and isinstance(input_info.func, _TensorInput) and (
                input_info.device_type == "NPU"):
            frozen_flag_list.append(1)
            logger.debug("No.%s arg is frozen data", idx)
        else:
            frozen_flag_list.append(0)
            logger.debug("No.%s arg is Data", idx)

    return frozen_flag_list


def frozen_data_by_constplaceholder(graph: GraphDef, frozen_flag_list: List, meta_outputs: Dict):
    frozen_data_op_list = []
    for op in graph.op:
        if op.type != "Data":
            continue
        if op.attr["index"].i < len(frozen_flag_list) and frozen_flag_list[op.attr["index"].i]:
            frozen_data_op_list.append(op)

    data_to_constplaceholder_name_mapping = {}
    for op in frozen_data_op_list:
        arg_idx = op.attr["index"].i
        name = f"ConstPlaceHolder_{arg_idx}"
        data_to_constplaceholder_name_mapping[op.name] = name
        with graph:
            constplaceholder = ge.ConstPlaceHolder(origin_shape=[], origin_format=2, storage_shape=[], storage_format=2,
                                                   expand_dim_rules="", dtype=0, addr=0, size=4, node_name=name)
            logger.debug('Replace %s by constructing fake %s [shape=[], dtype=DT_FLOAT, format=FORMAT_ND]', op.name,
                         name)
            constplaceholder.node.attr["update_node_from_fx_input_idx"].i = arg_idx
            if arg_idx in meta_outputs.keys() and meta_outputs[arg_idx] is not None:
                constplaceholder.set_meta(meta_outputs[arg_idx])

    update_op_input_name_from_mapping(graph, data_to_constplaceholder_name_mapping)


def _remove_op_controller_from_input(data: OpDef, op: OpDef):
    for idx, input_name in enumerate(op.input):
        if input_name.endswith("-1") and (input_name.split(":")[0] == data.name):
            logger.debug("remove ge graph op %s controller %s.", op.name, input_name)
            del op.input[idx]
            break


def remove_dead_data_and_reorder_data_index(graph: GraphDef):
    all_data_and_refcount: Dict[str, Tuple[OpDef, int]] = {}
    all_data_and_controller: Dict[str, List[OpDef]] = {}
    for op in graph.op:
        if op.type == "Data":
            all_data_and_refcount[GeTensor(op).tensor] = (op, 0)
            all_data_and_controller[op.name] = []
    for op in graph.op:
        if op.type == "Data":
            continue
        for _, op_in in enumerate(op.input):
            if op_in in all_data_and_refcount.keys():
                all_data_and_refcount[op_in] = (all_data_and_refcount[op_in][0], all_data_and_refcount[op_in][1] + 1)
            # record controller from Data
            input_name = op_in.split(":")[0]
            if op_in.endswith("-1") and (input_name in all_data_and_controller.keys()):
                all_data_and_controller[input_name].append(op)
    logger.info(f"before removing dead data, graph all inputs size={len(all_data_and_refcount)}.")

    saved_inputs_info = []
    to_del_data_name = []
    for _, op_tuple in all_data_and_refcount.items():
        if op_tuple[1] == 0:
            logger.debug("remove ge graph data op %s.", op_tuple[0].name)
            to_del_data_name.append(op_tuple[0].name)
            graph.op.remove(op_tuple[0])
            for op in all_data_and_controller.get(op_tuple[0].name, []):
                _remove_op_controller_from_input(op_tuple[0], op)
        else:
            logger.debug("update ge graph data index from %s to %s.", op_tuple[0].attr['index'].i,
                         len(saved_inputs_info))
            op_tuple[0].attr["index"].i = len(saved_inputs_info)
            if op_tuple[0].name not in graph.named_inputs_info.keys():
                raise AssertionError(f"Cannot find input info for data {op_tuple[0].name}. "
                                     f"All expected inputs info for data are {graph.named_inputs_info.keys()}.")
            saved_inputs_info.append(graph.named_inputs_info[op_tuple[0].name])
    for data_name in to_del_data_name:
        del graph.named_inputs_info[data_name]

    logger.info(f"after update index, graph all inputs size={len(saved_inputs_info)}.")
    return saved_inputs_info


_SIDE_EFFECT_OPS = {"PrintV2"}


def explict_order_for_side_effect_nodes(graph: GeGraph, graph_output_ref_input=None):
    """
    For nodes with side effects, such as nodes that write Data (allocate the same memory for its output as
    the graph input), and ge operators like Print with side effects (the order of screen printing needs to
    be guaranteed), we need to ensure that they are executed in the correct order.
    The following nodes will execute in strictly order:
    - Nodes that write graph input memory and all other users of the data node.
    - Nodes that have side effects, such as PrintV2.
    """
    strict_order_nodes: Dict[int, OpDef] = dict()  # Node to execute in strict order

    net_output: OpDef = None  # sink node for graph
    net_inputs: Dict[int, str] = {}  # net input index to op

    op_hint_order: Dict[str, int] = dict()  # op name to its hint order(origin eager order)
    name_to_op: Dict[str, OpDef] = {op.name: op for op in graph.op}
    for order, op in enumerate(graph.op):
        if op.type == "Data":
            net_inputs[op.attr["index"].i] = GeTensor(op).tensor
        elif op.type == "NetOutput":
            net_output = op
        elif op.type in _SIDE_EFFECT_OPS:
            logger.debug("Strict order find side effect op %s in order %s.", op.name, order)
            strict_order_nodes[order] = op
        op_hint_order[op.name] = order

    if len(strict_order_nodes) == 0:
        logger.debug("No side effect op found in graph, skip strict order optimization.")
        return

    strict_order_nodes = dict(sorted(strict_order_nodes.items()))
    ordered_nodes = list(strict_order_nodes.values())
    for i, op in enumerate(ordered_nodes[:-1]):
        dst: OpDef = ordered_nodes[i + 1]
        control = ControlTensor(op).controller
        logger.debug("Strict order add control edge from %s to %s.", op.name, dst.name)
        if control not in dst.input:
            dst.input.append(control)
    # Keep the last node connected to the net output to prevent side effect nodes pruned by ge
    sink_controller = ControlTensor(ordered_nodes[-1]).controller
    if ordered_nodes[-1].name not in [v.split(":")[0] for v in net_output.input]:
        net_output.input.append(sink_controller)

    graph_output_ref_input = graph_output_ref_input or {}
    input_to_its_written: Dict[str, str] = dict()
    net_output_src_nodes: List[str] = [t.split(":")[0] for t in net_output.input if not t.endswith(":-1")]
    for output_idx, op_name in enumerate(net_output_src_nodes):
        input_index = graph_output_ref_input.get(output_idx, None)
        if input_index is None:
            continue
        node = name_to_op[op_name]
        logger.debug("Strict order find node %s mutate input[%s] %s, "
                     "output index %s.", node.name, input_index, net_inputs[input_index], output_idx)
        input_to_its_written[net_inputs[input_index]] = node.name

    def get_op_input_other_users(op: OpDef):
        logger.debug("Strict order find node %s input %s.", op.name, op.input)
        users: Set[str] = set()
        for v in op.input:
            written_op = input_to_its_written.get(v, None)
            if written_op is not None:
                logger.debug("Strict order find node %s input %s is written by %s.", op.name, v, written_op)
                users.add(written_op)
                continue
            input_op = name_to_op[v.split(":")[0]]
            if op.type == "PrintV2" and input_op.type == "StringFormat":
                users |= get_op_input_other_users(input_op)
        return users

    for _, op in strict_order_nodes.items():
        other_users = get_op_input_other_users(op)
        for user in other_users:
            src, dst = op, name_to_op[user]
            if op_hint_order[src.name] > op_hint_order[dst.name]:
                src, dst = dst, src
            control = ControlTensor(src).controller
            logger.debug("Strict order add control edge from %s to %s.", src.name, dst.name)
            if control not in dst.input:
                dst.input.append(control)


def explict_order_for_cmo(graph: GeGraph):
    """
    Cmo nodes need be in strict order in graph.
    Cmo can be parallel to RefOp, system ensures that memory r/w are correct.
    """
    cmo_nodes: Dict[int, OpDef] = dict()  # Node to execute in strict order
    net_output: OpDef = None  # sink node for graph

    for order, op in enumerate(graph.op):
        if op.type == "NetOutput":
            net_output = op
        elif op.type == "Cmo":
            logger.debug("Strict order find Cmo in order %s.", order)
            cmo_nodes[order] = op

    if len(cmo_nodes) == 0:
        logger.debug("No Cmo found in graph, skip Cmo strict order optimization.")
        return

    cmo_nodes = dict(sorted(cmo_nodes.items()))
    ordered_nodes = list(cmo_nodes.values())
    for i, op in enumerate(ordered_nodes[:-1]):
        dst: OpDef = ordered_nodes[i + 1]
        control = ControlTensor(op).controller
        logger.debug("Strict order add control edge from %s to %s.", op.name, dst.name)
        if control not in dst.input:
            dst.input.append(control)
    # Keep the last node connected to the net output to prevent side effect nodes pruned by ge
    sink_controller = ControlTensor(ordered_nodes[-1]).controller
    if ordered_nodes[-1].name not in [v.split(":")[0] for v in net_output.input]:
        net_output.input.append(sink_controller)
