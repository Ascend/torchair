from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
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
from torchair._ge_concrete_graph.ge_converter.experimental.hcom_alltoall import _all_to_all_single


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


@register_fx_node_ge_converter(torch.ops.c10d_functional.wait_tensor.default)
def convert_c10d_functional_wait_tensor(
    self: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: Any = None,
):
    return ge.Identity(self)


if torch.__version__ >= '2.3.1':
    @register_fx_node_ge_converter(torch.ops._c10d_functional.wait_tensor.default)
    def convert_c10d_functional_wait_tensor_v2(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None,
    ):
        return ge.Identity(self)

    @register_fx_node_ge_converter(torch.ops._c10d_functional.all_reduce.default)
    def convert_c10d_functional_all_reduce_v2(
        self: Tensor,
        reduce_type: str,
        group_name: str,
        *,
        out: Tensor = None,
        meta_outputs: Any = None,
    ):
        group = torch.distributed.distributed_c10d._resolve_process_group(group_name)
        rank_list = torch.distributed.get_process_group_ranks(group)
        tag = torch.distributed.distributed_c10d._get_group_tag(group)
        hccl_group_name = get_group_name_and_record(tag, rank_list, len(rank_list))
        return ge.HcomAllReduce(self, reduction=normalize_reduceop_type(reduce_type), group=hccl_group_name, fusion=0)

    @register_fx_node_ge_converter(torch.ops._c10d_functional.all_gather_into_tensor.default)
    def convert_c10d_functional_all_gather_into_tensor_v2(
        self: Tensor,
        group_size: int,
        group_name: str,
        *,
        out: Tensor = None,
        meta_outputs: Any = None,
    ):
        group = torch.distributed.distributed_c10d._resolve_process_group(group_name)
        rank_list = torch.distributed.get_process_group_ranks(group)
        tag = torch.distributed.distributed_c10d._get_group_tag(group)
        hccl_group_name = get_group_name_and_record(tag, rank_list, group_size)
        return ge.HcomAllGather(self, group=hccl_group_name, rank_size=group_size)
    
    @register_fx_node_ge_converter(torch.ops._c10d_functional.reduce_scatter_tensor.default)
    def convert_c10d_functional_reduce_scatter_tensor_v2(
        self: Tensor,
        reduce_type: str,
        group_size: int,
        group_name: str,
        *,
        out: Tensor = None,
        meta_outputs: Any = None,
    ):
        group = torch.distributed.distributed_c10d._resolve_process_group(group_name)
        rank_list = torch.distributed.get_process_group_ranks(group)
        tag = torch.distributed.distributed_c10d._get_group_tag(group)
        hccl_group_name = get_group_name_and_record(tag, rank_list, group_size)
        return ge.HcomReduceScatter(self, reduction=normalize_reduceop_type(reduce_type),
                                    group=hccl_group_name, rank_size=group_size)

    @register_fx_node_ge_converter(torch.ops._c10d_functional.broadcast.default)
    def conveter_broadcast(
        input_tensor: torch.Tensor,
        src: int,
        group_name: str,
        *,
        out: Tensor = None,
        meta_outputs: Any = None,
    ):
        group = torch.distributed.distributed_c10d._resolve_process_group(group_name)
        rank_list = torch.distributed.get_process_group_ranks(group)
        tag = torch.distributed.distributed_c10d._get_group_tag(group)
        hccl_group_name = get_group_name_and_record(tag, rank_list, len(rank_list))
        return ge.HcomBroadcast([input_tensor], root_rank=src, group=hccl_group_name, fusion=0)[0]

    def decomp_c10d_functional_all_to_all_single(
        input_tensor: Tensor,
        output_split_sizes: Optional[List[int]],
        input_split_sizes: Optional[List[int]],
        group_name: str,
    ):
        group = torch.distributed.distributed_c10d._resolve_process_group(group_name)
        rank_list = torch.distributed.get_process_group_ranks(group)
        tag = torch.distributed.distributed_c10d._get_group_tag(group)
        return _all_to_all_single(input_tensor,
                                  output_split_sizes,
                                  input_split_sizes,
                                  tag,
                                  rank_list,
                                  len(rank_list))
