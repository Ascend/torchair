import copy
from typing import Any, Dict, List, Tuple, Union, Callable
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
            if op_in in host_data_ops.keys() and op.type != "Pack":
                host_data_ops[op_in] = (host_data_ops[op_in][0], host_data_ops[op_in][1] + 1)
            elif op_in in sym_pack_ops.keys():
                ops_with_pack_input[op_in].append((op, idx))
    for pack_op, peer_ops in ops_with_pack_input.items():
        for peer_op, peer_index in peer_ops:
            logger.debug(f"find pack op: {pack_op}, peer op: {GeTensor(peer_op).tensor}"
                         f", peer op input index: {peer_index}.")

    return host_data_ops, sym_pack_ops, ops_with_pack_input


def optimize_sym_pack(graph: GraphDef, inputs: List, placements: List, fx_inputs_mapping: Dict, fx_inputs_len: int):
    def _transfer_sym_pack_to_data():
        pack_fx_inputs: List[List[int]] = []
        ge_inputs_mapping: Dict = {value: key for key, value in fx_inputs_mapping.items()}
        for pack_name, pack_op in sym_pack_ops.items():
            with graph:
                data = ge.Data(index=len(inputs),
                               dtype=_ge_proto_dtype_to_ge_dtype(pack_op.input_desc[0].dtype),
                               shape=[len(pack_op.input)],
                               placement='CPU',
                               node_name=None)
                if GeTensor(pack_op)._meta is not None:
                    data.set_meta(GeTensor(pack_op)._meta)
                fx_inputs_mapping[fx_inputs_len + len(pack_fx_inputs)] = len(inputs)
                inputs.append(data)
                placements.append(Placement.HOST)
                pack_fx_inputs.append(
                    [ge_inputs_mapping[host_data_ops[pack_in][0].attr["index"].i] for pack_in in pack_op.input])
            logger.info(f"create data op {data.node.name} to replace pack op {pack_name}.")

            for pack_next_op, next_op_input_idx in ops_with_pack_input[GeTensor(pack_op).tensor]:
                pack_next_op.input[next_op_input_idx] = data.tensor
            graph.op.remove(pack_op)
        return pack_fx_inputs

    def _update_date_index_and_mapping():
        logger.info(f"before update index, graph input mapping {fx_inputs_mapping}.")
        ge_inputs_mapping: Dict = {value: key for key, value in fx_inputs_mapping.items()}
        saved_inputs, saved_placements = copy.copy(inputs), copy.copy(placements)
        inputs.clear()
        placements.clear()
        fx_inputs_mapping.clear()
        graph._indexed_inputs.clear()
        for input_idx, data_tensor in enumerate(saved_inputs):
            if data_tensor.tensor in host_data_ops.keys() and host_data_ops[data_tensor.tensor][1] == 0:
                logger.info(f"remove ge graph data op {data_tensor.tensor} and input mapping.")
                graph.op.remove(host_data_ops[data_tensor.tensor][0])
            else:
                data_tensor.node.attr["index"].i = len(inputs)
                graph._indexed_inputs[len(inputs)] = data_tensor.node
                fx_inputs_mapping[ge_inputs_mapping[input_idx]] = len(inputs)
                inputs.append(saved_inputs[input_idx])
                placements.append(saved_placements[input_idx])
        logger.info(f"after update index, graph input mapping {fx_inputs_mapping}.")

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
            all_fx_inputs.append([all_fx_inputs[idx] for idx in input_idx])
        return all_fx_inputs

    return pack_fx_input_for_data
