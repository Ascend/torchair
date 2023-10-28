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
    from torch.distributed.distributed_c10d import _world
    rank = torch.distributed.get_rank()
    ranklist = torch.distributed.get_process_group_ranks(_world.default_pg)
    y = ge.HcomAllReduce(self, reduction=normalize_reduceop_type(reduce_type), group="hccl_world_group", fusion=0)
    y._node.attr["ranklist"].list.i.extend(ranklist)
    try:
        hcom_info = _world.default_pg._get_backend(torch.device("npu")).get_hccl_comm(rank)
    except:
        logger.info(f'get_hccl_comm failed maybe in cpu export')
    else:
        y._node.attr["comm"].i = hcom_info
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
    from torch.distributed.distributed_c10d import _world
    rank = torch.distributed.get_rank()
    ranklist = torch.distributed.get_process_group_ranks(_world.default_pg)
    ranksize = len(ranklist)
    y = ge.HcomReduceScatter(self, reduction=normalize_reduceop_type(reduce_type),
                             group="hccl_world_group", rank_size=ranksize)
    y._node.attr["ranklist"].list.i.extend(ranklist)
    try:
        hcom_info = _world.default_pg._get_backend(torch.device("npu")).get_hccl_comm(rank)
    except:
        logger.info(f'get_hccl_comm failed maybe in cpu export')
    else:
        y._node.attr["comm"].i = hcom_info
    return y


@register_fx_node_ge_converter(torch.ops.c10d_functional.wait_tensor.default)
def convert_c10d_functional_wait_tensor(
    self: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: Any = None,
):
    return ge.Identity(self)
