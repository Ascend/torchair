from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable
import torch
import torch.distributed.distributed_c10d as c10d
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.hcom_utils import get_group_name_and_record
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
        rank_list: List[int],
        group_size: int,
        meta_outputs: Any = None,
):
    """broadcast(Tensor self, int src, str tag, int[] ranks, int group_size) -> Tensor"""
    group_name = get_group_name_and_record(tag, rank_list, group_size)
    return ge.HcomBroadcast([input_tensor], root_rank=src, group=group_name, fusion=0)[0]


def npu_broadcast_patch_dist(input_tensor, src, group=None, async_op=False):
    if not torch.distributed._functional_collectives._are_we_tracing():
        return torch.distributed.distributed_c10d.broadcast(input_tensor, src, group, async_op)
    if async_op:
        AssertionError(f'When you enable torch.compile or use the cache_compile feature, '
                       f'use the patch_for_hcom interface to ensure that collective communication functions '
                       f'are included in the graph. However, unlike the eager mode, the compile mode '
                       f'does not support the async_op = True parameter for collective communication APIs.')

    if group is None:
        group = c10d._world.default_pg
    ranks = torch.distributed.get_process_group_ranks(group)
    tag = c10d._get_group_tag(group)
    out = torch.ops.npu_define.broadcast(input_tensor, src, tag, ranks, len(ranks))
    input_tensor.copy_(out)


npu_define_lib.impl(op_broadcast, broadcast_meta, 'Meta')
npu_define_lib.impl(op_broadcast, broadcast_npu, 'PrivateUse1')
