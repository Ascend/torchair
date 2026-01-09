from typing import Any, List, Optional

import torch
import torch.distributed.distributed_c10d as c10d
from torch.fx.node import has_side_effect
from torch._C._distributed_c10d import ProcessGroup

from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.hcom_utils import get_group_name_and_record
from torchair.ge._ge_graph import Tensor, dont_prune_me

from .hcom_allreduce import npu_define_lib

# added the 'shape' parameter because dynamic shape is not supported
op_send = npu_define_lib.define(
    "_send(Tensor input_tensor, int? dst, int[] ranks, str pg_tag,"
    "int tag, int? group_dst=None, int[]? shape=None) -> None")


def send_npu(
        tensor: torch.Tensor,
        dst: Optional[int] = None,
        ranks: List[int] = None,
        pg_tag: str = "",
        tag: int = 0,
        group_dst: Optional[int] = None,
        shape: List[int] = None,
):
    group = c10d._find_or_create_pg_by_ranks_and_tag(pg_tag, ranks, len(ranks))
    c10d.send(tensor, dst, group, tag, group_dst=group_dst)
    return None


def send_meta(
        tensor: torch.Tensor,
        dst: Optional[int] = None,
        ranks: List[int] = None,
        pg_tag: str = "",
        tag: int = 0,
        group_dst: Optional[int] = None,
        shape: List[int] = None,
):
    return None


npu_define_lib.impl(op_send, send_meta, 'Meta')
npu_define_lib.impl(op_send, send_npu, 'PrivateUse1')


@register_fx_node_ge_converter(torch.ops.npu_define._send.default)
def convert_send(
        tensor: Tensor,
        dst: Optional[int] = None,
        ranks: List[int] = None,
        pg_tag: str = "",
        tag: int = 0,
        group_dst: Optional[int] = None,
        shape: List[int] = None,
        meta_outputs: Any = None,
):
    group_name = get_group_name_and_record(pg_tag, ranks, len(ranks))
    group = c10d._find_or_create_pg_by_ranks_and_tag(pg_tag, ranks, len(ranks))
    dest_rank = c10d._canonicalize_group_rank(group, dst, group_dst)
    ge.HcomSend(tensor, group=group_name, sr_tag=tag, dest_rank=dest_rank)


def npu_send_patch_dist(
        tensor: torch.Tensor,
        dst: Optional[int] = None,
        group: Optional[ProcessGroup] = None,
        tag: int = 0,
        group_dst: Optional[int] = None,
):
    if not torch.distributed._functional_collectives._are_we_tracing():
        return c10d.send(tensor, dst, group, tag, group_dst)
    
    if group is None:
        group = c10d._world.default_pg
    rank_list = torch.distributed.get_process_group_ranks(group)
    pg_tag = c10d._get_group_tag(group)

    # used for static shape
    shape = tensor.shape
    torch.ops.npu_define._send(tensor, dst, rank_list, pg_tag, tag, group_dst, shape)


has_side_effect(torch.ops.npu_define._send.default)


# added the 'shape' parameter because dynamic shape is not supported
op_recv = npu_define_lib.define(
    "_recv(Tensor out_tensor, int? src, int[] ranks, str pg_tag,"
    "int tag, int? group_src=None, int[]? shape=None) -> Tensor")


def recv_npu(
        output_tensor: torch.Tensor,
        src: Optional[int] = None,
        ranks: List[int] = None,
        pg_tag: str = "",
        tag: int = 0,
        group_src: Optional[int] = None,
        shape: List[int] = None,
):
    pg = c10d._find_or_create_pg_by_ranks_and_tag(pg_tag, ranks, len(ranks))
    c10d.recv(output_tensor, src, group=pg, tag=tag, group_src=group_src)
    return output_tensor


def recv_meta(
        tensor: torch.Tensor,
        src: Optional[int] = None,
        ranks: List[int] = None,
        pg_tag: str = "",
        tag: int = 0,
        group_src: Optional[int] = None,
        shape: List[int] = None,
):
    out_size = list(tensor.size())
    return tensor.new_empty(out_size)


npu_define_lib.impl(op_recv, recv_meta, 'Meta')
npu_define_lib.impl(op_recv, recv_npu, 'PrivateUse1')


@register_fx_node_ge_converter(torch.ops.npu_define._recv.default)
def convert_recv(
        tensor: Tensor,
        src: Optional[int] = None,
        ranks: List[int] = None,
        pg_tag: str = "",
        tag: int = 0,
        group_src: Optional[int] = None,
        shape: List[int] = None,
        meta_outputs: Any = None,
):
    group_name = get_group_name_and_record(pg_tag, ranks, len(ranks))
    group = c10d._find_or_create_pg_by_ranks_and_tag(pg_tag, ranks, len(ranks))
    src_rank = c10d._canonicalize_group_rank(group, src, group_src)
    op = ge.HcomReceive(group=group_name, src_rank=src_rank, sr_tag=tag, shape=shape, dtype=tensor.dtype)
    dont_prune_me(op)
    return op


def npu_recv_patch_dist(
        tensor: torch.Tensor,
        src: Optional[int] = None,
        group: Optional[ProcessGroup] = None,
        tag: int = 0,
        group_src: Optional[int] = None,
):
    if not torch.distributed._functional_collectives._are_we_tracing():
        return c10d.recv(tensor, src, group, tag, group_src)
    
    if group is None:
        group = c10d._world.default_pg
    rank_list = torch.distributed.get_process_group_ranks(group)
    pg_tag = c10d._get_group_tag(group)

    # used for static shape
    shape = tensor.shape
    out = torch.ops.npu_define._recv(tensor, src, rank_list, pg_tag, tag, group_src, shape)
    tensor.copy_(out)


has_side_effect(torch.ops.npu_define._recv.default)
