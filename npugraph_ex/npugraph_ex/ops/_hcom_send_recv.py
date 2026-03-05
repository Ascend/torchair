from typing import List, Optional

import torch
import torch.distributed.distributed_c10d as c10d
from torch.fx.node import has_side_effect
from torch._C._distributed_c10d import ProcessGroup

from ._npu_define_lib import npu_define_lib


# added the 'shape' parameter because dynamic shape is not supported
if not hasattr(getattr(torch.ops, "npu_define"), "_send"):
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

    has_side_effect(torch.ops.npu_define._send.default)


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


# added the 'shape' parameter because dynamic shape is not supported
if not hasattr(getattr(torch.ops, "npu_define"), "_recv"):
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

    has_side_effect(torch.ops.npu_define._recv.default)


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
