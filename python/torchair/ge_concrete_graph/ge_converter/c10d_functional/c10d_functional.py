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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.core.utils import logger
from torchair.ge_concrete_graph.utils import normalize_reduceop_type


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


@register_fx_node_ge_converter(torch.ops.c10d_functional.wait_tensor.default)
def convert_c10d_functional_wait_tensor(
    self: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: Any = None,
):
    return ge.Identity(self)
