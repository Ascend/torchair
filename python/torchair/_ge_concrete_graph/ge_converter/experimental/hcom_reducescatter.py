from typing import Any, List, Optional

import torch
import torch.distributed.distributed_c10d as c10d
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.hcom_utils import get_group_name_and_record
from torchair._ge_concrete_graph.utils import dtype_promote, normalize_reduceop_type
from torchair.ge._ge_graph import Tensor, DataType, dont_prune_me

from .hcom_allreduce import npu_define_lib, convert_reduce_op

op_reduce_scatter_tensor_uneven = npu_define_lib.define(
    "reduce_scatter_tensor_uneven(Tensor input_tensor, SymInt[] send_counts, SymInt recv_count, \
    str reduce_type, str tag, int[] rank_list, int group_size, SymInt[] send_displacements) -> Tensor")


def reduce_scatter_tensor_uneven_meta(
        input_tensor: torch.Tensor,
        send_counts: List[int],
        recv_count: int,
        reduce_type: str,
        tag: str,
        rank_list: List,
        group_size: int,
        send_displacements: List[int]
):
    out_size = list(input_tensor.size())
    out_size[0] = recv_count
    return input_tensor.new_empty(out_size)


npu_define_lib.impl(op_reduce_scatter_tensor_uneven, reduce_scatter_tensor_uneven_meta, 'Meta')


@register_fx_node_ge_converter(torch.ops.npu_define.reduce_scatter_tensor_uneven.default)
def convert_reduce_scatter_tensor_uneven(
        input_tensor: Tensor,
        send_counts: List[int],
        recv_count: int,
        reduce_type: str,
        tag: str,
        rank_list: List,
        group_size: int,
        send_displacements: List[int],
        meta_outputs: Any = None,
):
    send_counts, send_displacements, recv_count = dtype_promote(send_counts,
                                                                send_displacements, recv_count,
                                                                target_dtype=DataType.DT_INT64)
    group_name = get_group_name_and_record(tag, rank_list, group_size)
    op = ge.HcomReduceScatterV(input_tensor, send_counts=send_counts, send_displacements=send_displacements,
                               reduction=normalize_reduceop_type(reduce_type), recv_count=recv_count,
                               group=group_name)
    # send_counts could be 0
    dont_prune_me(op)
    return op


def npu_reduce_scatter_tensor_uneven_patch_dist(
        output_tensor,
        input_tensor,
        input_split_sizes=None,
        op=torch.distributed.ReduceOp.SUM,
        group=None,
        async_op=False,
):
    if not torch.distributed._functional_collectives._are_we_tracing():
        from torchair import REDUCE_SCATTER_TENSOR_UNEVEN
        if REDUCE_SCATTER_TENSOR_UNEVEN is None:
            raise AttributeError(f'torch_npu.distributed has no attribute: reduce_scatter_tensor_uneven')
        REDUCE_SCATTER_TENSOR_UNEVEN(output_tensor, input_tensor, input_split_sizes, op, group, async_op)
    if async_op:
        raise AssertionError(f'When you enable torch.compile or use the cache_compile feature, '
                             f'use the patch_for_hcom interface to ensure that collective communication functions '
                             f'are included in the graph. However, unlike the eager mode, the compile mode '
                             f'does not support the async_op = True parameter for collective communication APIs.')
    reduce_type = convert_reduce_op(op)
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

    npu_output = torch.ops.npu_define.reduce_scatter_tensor_uneven(input_tensor, send_counts=input_split_sizes,
                                                                   recv_count=output_tensor.size(0),
                                                                   reduce_type=reduce_type,
                                                                   tag=tag,
                                                                   rank_list=rank_list, group_size=group_size,
                                                                   send_displacements=[])
    output_tensor.copy_(npu_output.reshape(output_tensor.size()))
