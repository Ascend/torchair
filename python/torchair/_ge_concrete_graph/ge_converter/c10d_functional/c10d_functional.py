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
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.core.utils import logger
from torchair._ge_concrete_graph.utils import normalize_reduceop_type, dtype_promote
from torchair._ge_concrete_graph.hcom_utils import get_group_name_and_record


try:
    ALL_TO_ALL_SINGLE = torch.ops.c10d_functional.all_to_all_single.default
except Exception:
    ALL_TO_ALL_SINGLE = None


@register_fx_node_ge_converter(torch.ops.c10d_functional.all_reduce.default)
def convert_c10d_functional_all_reduce(
    self: Tensor,
    reduce_type: str,
    tag: str,
    rank_list: List,
    group_size: int,
    *,
    out: Tensor = None,
    meta_outputs: Any = None,
):
    group_name = get_group_name_and_record(tag, rank_list, group_size)
    return ge.HcomAllReduce(self, reduction=normalize_reduceop_type(reduce_type), group=group_name, fusion=0)


@register_fx_node_ge_converter(torch.ops.c10d_functional.reduce_scatter_tensor.default)
def convert_c10d_functional_reduce_scatter_tensor(
    self: Tensor,
    reduce_type: str,
    tag: str,
    rank_list: List,
    group_size: int,
    *,
    out: Tensor = None,
    meta_outputs: Any = None,
):
    group_name = get_group_name_and_record(tag, rank_list, group_size)
    return ge.HcomReduceScatter(self, reduction=normalize_reduceop_type(reduce_type),
                                group=group_name, rank_size=group_size)


@register_fx_node_ge_converter(torch.ops.c10d_functional.all_gather_into_tensor.default)
def convert_c10d_functional_all_gather_into_tensor(
    self: Tensor,
    tag: str,
    rank_list: List,
    group_size: int,
    *,
    out: Tensor = None,
    meta_outputs: Any = None,
):
    group_name = get_group_name_and_record(tag, rank_list, group_size)
    return ge.HcomAllGather(self, group=group_name, rank_size=group_size)


@register_fx_node_ge_converter(ALL_TO_ALL_SINGLE)
def convert_c10d_functional_all_to_all_single(
    input_tensor: Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
    tag: str,
    rank_list: List,
    group_size: int,
    *,
    out: Tensor = None,
    meta_outputs: Any = None,
):
    if input_split_sizes is None and output_split_sizes is None:
        input_split_sizes = []
        output_split_sizes = []
        spilt_size = input_tensor.get_numel() // group_size
        for i in range(group_size):
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

    group_name = get_group_name_and_record(tag, rank_list, group_size)
    y = ge.HcomAllToAllV(send_data=input_tensor, send_counts=send_counts, send_displacements=send_displacements,
                         recv_counts=recv_counts, recv_displacements=recv_displacements, group=group_name)
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
