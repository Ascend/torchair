from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)
import torch
import torch.distributed._functional_collectives
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, DataType
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.core.utils import logger
from torchair.ge_concrete_graph.utils import normalize_reduceop_type, dtype_promote


try:
    ALL_TO_ALL_SINGLE = torch.ops.c10d_functional.all_to_all_single.default
except Exception:
    ALL_TO_ALL_SINGLE = None


@register_fx_node_ge_converter(torch.ops.c10d_functional.all_reduce.default)
def convert_c10d_functional_all_reduce(
    self: Tensor,
    reduce_type: str,
    tag: str,
    ranklist: List,
    groupsize: int,
    *,
    out: Tensor = None,
    meta_outputs: Any = None,
):
    rank = torch.distributed.get_rank()
    pg = torch.distributed.distributed_c10d._find_or_create_pg_by_ranks_and_tag(tag, ranklist, groupsize)
    device = torch.distributed.distributed_c10d._get_pg_default_device(pg)
    if device.type == "cpu":
        y = ge.HcomAllReduce(self, reduction=normalize_reduceop_type(reduce_type),
                             group=tag, fusion=0)
        logger.debug(f'c10d_functional.all_reduce convert in cpu export, group name set as tag name')
    elif device.type == "npu":
        hcom_name = pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        y = ge.HcomAllReduce(self, reduction=normalize_reduceop_type(reduce_type),
                             group=hcom_name, fusion=0)
    else:
        raise ValueError("The initialized aggregate communication backend is not a CPU or NPU.")
    y._node.attr["ranklist"].list.i.extend(ranklist)
    return y


@register_fx_node_ge_converter(torch.ops.c10d_functional.reduce_scatter_tensor.default)
def convert_c10d_functional_reduce_scatter_tensor(
    self: Tensor,
    reduce_type: str,
    tag: str,
    ranklist: List,
    groupsize: int,
    *,
    out: Tensor = None,
    meta_outputs: Any = None,
):
    rank = torch.distributed.get_rank()
    pg = torch.distributed.distributed_c10d._find_or_create_pg_by_ranks_and_tag(tag, ranklist, groupsize)
    device = torch.distributed.distributed_c10d._get_pg_default_device(pg)
    if device.type == "cpu":
        y = ge.HcomReduceScatter(self, reduction=normalize_reduceop_type(reduce_type),
                                 group=tag, rank_size=groupsize)
        logger.debug(f'c10d_functional.all_reduce convert in cpu export, group name set as tag name')
    elif device.type == "npu":
        hcom_name = pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        y = ge.HcomReduceScatter(self, reduction=normalize_reduceop_type(reduce_type),
                                 group=hcom_name, rank_size=groupsize)
    else:
        raise ValueError("The initialized aggregate communication backend is not a CPU or NPU.")
    y._node.attr["ranklist"].list.i.extend(ranklist)
    return y


@register_fx_node_ge_converter(torch.ops.c10d_functional.all_gather_into_tensor.default)
def convert_c10d_functional_all_gather_into_tensor(
    self: Tensor,
    tag: str,
    ranklist: List,
    groupsize: int,
    *,
    out: Tensor = None,
    meta_outputs: Any = None,
):
    rank = torch.distributed.get_rank()
    pg = torch.distributed.distributed_c10d._find_or_create_pg_by_ranks_and_tag(tag, ranklist, groupsize)
    device = torch.distributed.distributed_c10d._get_pg_default_device(pg)
    if device.type == "cpu":
        y = ge.HcomAllGather(self, group=tag, rank_size=groupsize)
        logger.debug(f'c10d_functional.all_gather_into_tensor convert in cpu export, group name set as tag name')
    elif device.type == "npu":
        hcom_name = pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        y = ge.HcomAllGather(self, group=hcom_name, rank_size=groupsize)
    else:
        raise ValueError("The initialized aggregate communication backend is not a CPU or NPU.")
    y._node.attr["ranklist"].list.i.extend(ranklist)
    return y


@register_fx_node_ge_converter(ALL_TO_ALL_SINGLE)
def convert_c10d_functional_all_to_all_single(
    input_tensor: Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
    tag: str,
    ranklist: List,
    groupsize: int,
    *,
    out: Tensor = None,
    meta_outputs: Any = None,
):
    rank = torch.distributed.get_rank()
    pg = torch.distributed.distributed_c10d._find_or_create_pg_by_ranks_and_tag(tag, ranklist, groupsize)
    device = torch.distributed.distributed_c10d._get_pg_default_device(pg)

    if input_split_sizes is None and output_split_sizes is None:
        input_split_sizes = []
        output_split_sizes = []
        spilt_size = input_tensor.get_numel() // groupsize
        for i in range(groupsize):
            input_split_sizes.append(spilt_size)
            output_split_sizes.append(spilt_size)

    send_counts = []
    recv_counts = []
    send_displacements = [0]
    recv_displacements = [0]
    if len(input_split_sizes) != len(output_split_sizes):
        raise AssertionError
    for i, split_size in enumerate(input_split_sizes):
        send_counts.append(split_size)
        if i > 0:
            send_displacements.append(send_displacements[i - 1] + send_counts[i - 1])

    for i, split_size in enumerate(output_split_sizes):
        recv_counts.append(split_size)
        if i > 0:
            recv_displacements.append(recv_displacements[i - 1] + recv_counts[i - 1])
        
    send_counts, send_displacements, recv_counts, recv_displacements = dtype_promote(send_counts, 
        send_displacements, recv_counts, recv_displacements, target_dtype=DataType.DT_INT64)
            
    if device.type == "cpu":
        y = ge.HcomAllToAllV(send_data=input_tensor, send_counts=send_counts, send_displacements=send_displacements,
                             recv_counts=recv_counts, recv_displacements=recv_displacements, group=tag)
        logger.debug(f'c10d_functional.all_to_all_single convert in cpu export, group name set as tag name')
    elif device.type == "npu":
        hcom_name = pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        y = ge.HcomAllToAllV(send_data=input_tensor, send_counts=send_counts, send_displacements=send_displacements,
                             recv_counts=recv_counts, recv_displacements=recv_displacements, group=hcom_name)
    else:
        raise ValueError("The initialized aggregate communication backend is not a CPU or NPU.")
    y._node.attr["ranklist"].list.i.extend(ranklist)
    output_tensor = ge.Reshape(y, shape=meta_outputs.size)
    return output_tensor
    

@register_fx_node_ge_converter(torch.ops.c10d_functional.wait_tensor.default)
def convert_c10d_functional_wait_tensor(
    self: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: Any = None,
):
    return ge.Identity(self)