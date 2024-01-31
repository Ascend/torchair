import copy
from collections import namedtuple
from typing import Any, Dict, List, Tuple, Union, Callable
import torch
from torchair.core.utils import logger
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef, TensorDescriptor, TensorDef, OpDef
from torchair.ge_concrete_graph.ge_graph import _ge_proto_dtype_to_ge_dtype
from torchair.ge_concrete_graph.ge_graph import Tensor as GeTensor
from torchair.ge_concrete_graph.utils import Placement

from . import ge_apis as ge


def _find_packs_and_io_ops(graph: GraphDef):
    host_data_ops: Dict[str, Tuple[OpDef, int]] = {}
    sym_pack_ops: Dict[str, OpDef] = {}
    for op in graph.op:
        if op.type == "Data" and op.input_desc[0].device_type == "CPU":
            host_data_ops[GeTensor(op).tensor] = (op, 0)
        elif op.type == "Pack" and op.attr['_inputs_all_sym'].b == True:
            assert op.attr["axis"].i == 0, f"sym pack op attr axis must be 0."
            sym_pack_ops[GeTensor(op).tensor] = op
    logger.info(f"find all host data ops {host_data_ops.keys()}, and sym pack ops {sym_pack_ops.keys()}.")

    ops_with_pack_input: Dict[str, List[Tuple[OpDef, int]]] = {pack_name: [] for pack_name in sym_pack_ops.keys()}
    for op in graph.op:
        if op.type == "Data":
            continue
        for idx, op_in in enumerate(op.input):
            if op_in in host_data_ops.keys() and (op.type != "Pack" or GeTensor(op).tensor not in sym_pack_ops.keys()):
                host_data_ops[op_in] = (host_data_ops[op_in][0], host_data_ops[op_in][1] + 1)
            elif op_in in sym_pack_ops.keys():
                ops_with_pack_input[op_in].append((op, idx))
    for pack_op, peer_ops in ops_with_pack_input.items():
        for peer_op, peer_index in peer_ops:
            logger.debug(f"find pack op: {pack_op}, peer op: {GeTensor(peer_op).tensor}"
                         f", peer op input index: {peer_index}.")

    return host_data_ops, sym_pack_ops, ops_with_pack_input


def _find_data_of_same_pack(datas: List[Tuple[GeTensor, OpDef]], pack_op: OpDef):
    for data_tensor, recorded_pack in datas:
        if len(recorded_pack.input) != len(pack_op.input):
            continue
        if all([recorded_pack.input[i] == pack_op.input[i] for i in range(len(recorded_pack.input))]):
            return data_tensor
    return None


def optimize_sym_pack(graph: GraphDef, inputs: List, placements: List, fx_inputs_mapping: Dict, fx_inputs_len: int):
    def _transfer_sym_pack_to_data():
        pack_fx_inputs: List[List[int]] = []
        additional_data: List[Tuple[GeTensor, OpDef]] = []
        ge_inputs_mapping: Dict = {value: key for key, value in fx_inputs_mapping.items()}
        for pack_name, pack_op in sym_pack_ops.items():
            new_data = _find_data_of_same_pack(additional_data, pack_op)
            if new_data is None:
                with graph:
                    new_data = ge.Data(index=len(inputs),
                                       dtype=_ge_proto_dtype_to_ge_dtype(pack_op.input_desc[0].dtype),
                                       shape=[len(pack_op.input)],
                                       placement='CPU',
                                       node_name=None)
                if GeTensor(pack_op)._meta is not None:
                    new_data.set_meta(GeTensor(pack_op)._meta)
                fx_inputs_mapping[fx_inputs_len + len(pack_fx_inputs)] = len(inputs)
                inputs.append(new_data)
                placements.append(Placement.HOST)

                pack_fx_inputs.append(
                    [ge_inputs_mapping[host_data_ops[pack_in][0].attr["index"].i] for pack_in in pack_op.input])
                additional_data.append((new_data, pack_op))
                logger.debug(f"create new data op {new_data.node.name} to replace pack op {pack_name}.")
            else:
                logger.debug(f"skip create data, set data op {new_data.node.name} to replace pack op {pack_name}.")

            for pack_next_op, next_op_input_idx in ops_with_pack_input[GeTensor(pack_op).tensor]:
                pack_next_op.input[next_op_input_idx] = new_data.tensor
            graph.op.remove(pack_op)
        return pack_fx_inputs

    def _update_date_index_and_mapping():
        logger.info(f"before update index, graph input num = {len(inputs)}, input mapping {fx_inputs_mapping}.")
        ge_inputs_mapping: Dict = {value: key for key, value in fx_inputs_mapping.items()}
        saved_inputs, saved_placements = copy.copy(inputs), copy.copy(placements)
        inputs.clear()
        placements.clear()
        fx_inputs_mapping.clear()
        saved_indexed_input_ops = copy.copy(graph.indexed_inputs())
        graph.indexed_inputs().clear()
        for input_idx, data_tensor in enumerate(saved_inputs):
            if data_tensor.tensor in host_data_ops.keys() and host_data_ops[data_tensor.tensor][1] == 0:
                logger.debug(f"remove ge graph data op {data_tensor.tensor} and input mapping.")
                graph.op.remove(host_data_ops[data_tensor.tensor][0])
            else:
                data_tensor.node.attr["index"].i = len(inputs)
                graph.indexed_inputs()[len(inputs)] = data_tensor.node
                fx_inputs_mapping[ge_inputs_mapping[input_idx]] = len(inputs)
                inputs.append(saved_inputs[input_idx])
                placements.append(saved_placements[input_idx])
        for idx in range(len(saved_inputs), len(saved_indexed_input_ops)):
            saved_indexed_input_ops[idx].attr["index"].i = graph.num_inputs()
            graph.indexed_inputs()[len(inputs)] = saved_indexed_input_ops[idx]
        logger.info(f"after update index, graph input num = {len(inputs)}, input mapping {fx_inputs_mapping}.")

    logger.info(f"before pack optimize, graph fx inputs size {fx_inputs_len}, graph ge inputs size {len(inputs)},"
                f", graph input mapping {fx_inputs_mapping}")
    assert len(placements) == len(
        inputs), f"graph inputs size {len(inputs)} is not equal to input_placements size {len(placements)}"
    assert len(fx_inputs_mapping) == len(
        inputs), f"graph inputs size {len(inputs)} is not equal to input_mappings size {len(fx_inputs_mapping)}"

    host_data_ops, sym_pack_ops, ops_with_pack_input = _find_packs_and_io_ops(graph)
    additional_fx_inputs: List[List[int]] = _transfer_sym_pack_to_data()
    _update_date_index_and_mapping()

    def pack_fx_input_for_data(*args: Any):
        all_fx_inputs = list(args)
        for input_idx in additional_fx_inputs:
            all_fx_inputs.append(torch.tensor([all_fx_inputs[idx] for idx in input_idx]))
        return all_fx_inputs

    return pack_fx_input_for_data


ref_op_info = namedtuple("ref_op_info", ["op_def", "output_id", "ref_input_name", "output_ref_count", "is_net_output"])


def _get_output_to_input_ref_idx(op: OpDef) -> Dict[int, int]:
    """
    If there are more inplace op that need to be optimized similarly, it is necessary to
    add a mapping relationship between output and references input in this function.
    """
    ref_idx_mapping: Dict[int, int] = {}
    if op.type == "Scatter":
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
        for idx, op_in in enumerate(op.input):
            if op_in in data_output_count.keys():
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
            logger.debug(f"Assign op: {assign_tensor} is not used to update ref_op output to data op.")
            continue
        ref_op, ref_idx, ref_op_input = ref_op_infos[assign_op.input[1]].op_def, \
                                        ref_op_infos[assign_op.input[1]].output_id, \
                                        ref_op_infos[assign_op.input[1]].ref_input_name
        if ref_op_input not in tensormove_ops.keys():
            logger.debug(f"ref_op input: {ref_op_input} type is not TensorMove, skip TensorMove optimization.")
            continue
        tensormove_op = tensormove_ops[ref_op_input]
        if tensormove_op.input[0] != assign_op.input[0]:
            logger.debug(f"TensorMove input: {tensormove_op.input[0]} is not Data to be assigned, skip optimization.")
            continue
        data_output_count = data_ops[assign_op.input[0]][1]
        if (data_output_count != 2):
            logger.debug(f"Data op: {assign_op.input[0]} output count {data_output_count} is not equal to two, "
                         f"which means the current Data op cannot be modified inplace, skip TensorMove optimization.")
            continue

        ref_out_count, is_net_output = ref_op_infos[assign_op.input[1]].output_ref_count, \
                                       ref_op_infos[assign_op.input[1]].is_net_output
        if is_net_output or ref_out_count == 1:
            logger.debug(f"Assign op: {assign_tensor} is used to copy {assign_op.input[1]} to {assign_op.input[0]}, "
                         f"update ref_op ref_input_{ref_idx} from {ref_op_input} to {tensormove_op.input[0]}.")
            logger.debug(f"Ref_op ref_output_{ref_idx} is a net_output, Assign op can not be removed, "
                         f"only remove TensorMove op: {ref_op_input}.")
            ref_op.input[ref_idx] = tensormove_op.input[0]
            graph.op.remove(tensormove_op)
        else:
            logger.debug(f"Assign op: {assign_tensor} is used to copy {assign_op.input[1]} to {assign_op.input[0]}, "
                         f"update ref_op ref_input_{ref_idx} from {ref_op_input} to {tensormove_op.input[0]}, "
                         f"and remove Assign op: {assign_tensor} and TensorMove op: {ref_op_input}.")
            ref_op.input[ref_idx] = tensormove_op.input[0]
            graph.op.remove(assign_op)
            graph.op.remove(tensormove_op)

    return ref_data_idx
