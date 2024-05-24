from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable
import warnings
import torch
from torch.library import Library
from torch._decomp import register_decomposition
import torch.distributed.distributed_c10d as c10d
from torchair.ge_concrete_graph.ge_graph import Tensor, DataType
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.utils import normalize_reduceop_type, dtype_promote
from torchair.core.utils import logger
from .hcom_allreduce import npu_define_lib

op_broadcast = npu_define_lib.define(
    "broadcast(Tensor self, int src, str tag, int[] ranks, int group_size) -> Tensor")


def broadcast_npu(
        input_tensor: torch.Tensor,
        src: int,
        tag: str,
        ranks: List[int],
        group_size: int, ):
    pg = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    c10d.broadcast(input_tensor, src, group=pg, async_op=False)
    return input_tensor


def broadcast_meta(
        input_tensor: torch.Tensor,
        src: int,
        tag: str,
        ranks: List[int],
        group_size: int, ):
    return torch.empty_like(input_tensor)


@register_fx_node_ge_converter(torch.ops.npu_define.broadcast.default)
def conveter_broadcast(
        input_tensor: torch.Tensor,
        src: int,
        tag: str,
        ranks: List[int],
        group_size: int,
        meta_outputs: Any = None,
):
    """broadcast(Tensor self, int src, str tag, int[] ranks, int group_size) -> Tensor"""
    from torch.distributed.distributed_c10d import _world
    device = torch.distributed.distributed_c10d._get_pg_default_device(_world.default_pg)
    if device.type == "cpu":
        y = ge.HcomBroadcast([input_tensor], root_rank=src, group="hccl_world_group", fusion=0)
        logger.debug(f'npu_define.broadcast convert in cpu export')
    elif device.type == "npu":
        rank = torch.distributed.get_rank()
        hcom_name = _world.default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        y = ge.HcomBroadcast([input_tensor], root_rank=src, group=hcom_name, fusion=0)
    else:
        raise ValueError("The initialized aggregate communication backend is not a CPU or NPU.")
    y[0]._node.attr["ranklist"].list.i.extend(ranks)
    return y[0]


def npu_broadcast_patch_dist(input_tensor, src, group=None, async_op=False):
    if group is None:
        group = c10d._world.default_pg
    ranks = torch.distributed.get_process_group_ranks(group)
    tag = c10d._get_group_tag(group)
    out = torch.ops.npu_define.broadcast(input_tensor, src, tag, ranks, len(ranks))
    input_tensor.copy_(out)


npu_define_lib.impl(op_broadcast, broadcast_meta, 'Meta')
npu_define_lib.impl(op_broadcast, broadcast_npu, 'PrivateUse1')
torch.distributed.broadcast = npu_broadcast_patch_dist
