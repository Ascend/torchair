from typing import Any, List, Optional, Union

import torch
import torch.distributed.distributed_c10d as c10d
from torch._decomp import register_decomposition

from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.hcom_utils import get_group_name_and_record
from torchair._ge_concrete_graph.utils import dtype_promote, normalize_reduceop_type
from torchair.ge._ge_graph import Tensor, DataType, is_sym
from .hcom_allreduce import npu_define_lib, convert_reduce_op


op_reduce_scatter_tensor_uneven = npu_define_lib.define(
    "reduce_scatter_tensor_uneven(Tensor input, SymInt[] send_counts, SymInt[] send_displacements, \
     SymInt recv_count, str op, str tag, int[] ranks, int group_size) -> Tensor")


def reduce_scatter_tensor_uneven_meta(
        input_tensor: torch.Tensor,
        send_counts: List[int],
        send_displacements: List[int],
        recv_count: int,
        reduce_type: str,
        tag: str,
        ranklist: List,
        group_size: int,
):
    out_size = list(input_tensor.size())
    out_size[0] = recv_count // input_tensor[0].numel()
    return input_tensor.new_empty(out_size)


npu_define_lib.impl(op_reduce_scatter_tensor_uneven, reduce_scatter_tensor_uneven_meta, 'Meta')


@register_fx_node_ge_converter(torch.ops.npu_define.reduce_scatter_tensor_uneven.default)
def convert_reduce_scatter_tensor_uneven(
        self: Tensor,
        send_counts: List[int],
        send_displacements: List[int],
        recv_count: int,
        reduce_type: str,
        tag: str,
        rank_list: List,
        group_size: int,
        meta_outputs: Any = None,
):
    for dim in self.symsize:
        if is_sym(dim):
            raise NotImplementedError("No support dynamic: reduce_scatter_tensor_uneven")
    send_counts, send_displacements, recv_count = dtype_promote(send_counts,
                                                                send_displacements, recv_count,
                                                                target_dtype=DataType.DT_INT64)
    group_name = get_group_name_and_record(tag, rank_list, group_size)
    return ge.HcomReduceScatterV(self, send_counts=send_counts, send_displs=send_displacements,
                                 reduction=normalize_reduceop_type(reduce_type), recv_count=recv_count,
                                 group=group_name)


def npu_reduce_scatter_tensor_uneven_patch_dist(
        output_tensor,
        input_tensor,
        input_split_sizes=None,
        op=torch.distributed.ReduceOp.SUM,
        group=None,
        async_op=False,
):
    if not torch.distributed._functional_collectives._are_we_tracing():
        import torch_npu
        torch_npu.distributed.reduce_scatter_tensor_uneven(output_tensor, input_tensor, input_split_sizes, op,
                                                           group, async_op)
    if async_op:
        raise AssertionError(f'When you enable torch.compile or use the cache_compile feature, '
                       f'use the patch_for_hcom interface to ensure that collective communication functions '
                       f'are included in the graph. However, unlike the eager mode, the compile mode '
                       f'does not support the async_op = True parameter for collective communication APIs.')

    op = convert_reduce_op(op)
    if group is None:
        group = c10d._world.default_pg
    rank_list = torch.distributed.get_process_group_ranks(group)
    tag = c10d._get_group_tag(group)
    group_size = len(rank_list)

    # each rank get equal split from input tensor by dim 0
    if input_split_sizes is None:
        input_split_sizes = [input_tensor.size(0) // group_size for _ in rank_list]
    if sum(input_split_sizes) != input_tensor.size(0):
        raise AssertionError(f'Split sizes sum does not match total dim 0 size')
    input_row_size = input_tensor.numel() // input_tensor.size(0) if input_tensor.size(0) != 0 else 1
    send_counts = []
    recv_count = output_tensor.numel()
    send_displacements = [0]
    for i, split_size in enumerate(input_split_sizes):
        send_counts.append(split_size * input_row_size)
        if i > 0:
            send_displacements.append(send_displacements[i - 1] + send_counts[i - 1])
    npu_output = torch.ops.npu_define.reduce_scatter_tensor_uneven(input_tensor, send_counts, send_displacements,
                                                                   recv_count, op, tag, rank_list, group_size)
    output_tensor.copy_(npu_output.reshape(output_tensor.size()))
