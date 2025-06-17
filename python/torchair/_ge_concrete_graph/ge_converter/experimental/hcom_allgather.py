from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable

import torch
import torch.distributed.distributed_c10d as c10d
from torch._decomp import register_decomposition
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.hcom_utils import get_group_name_and_record
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge._ge_graph import Tensor, DataType

from .hcom_allreduce import npu_define_lib
from .hcom_broadcast import op_broadcast

op_allgather = npu_define_lib.define(
    "allgather(Tensor[] tensor_list,Tensor input, str tag, int[] ranks, int group_size) -> Tensor[]")

op_allgather_in_tensor = npu_define_lib.define(
    "allgather_in_tensor(Tensor out, Tensor input, str tag, int[] ranks, int group_size) -> Tensor")

_op_allgather_in_tensor_shape = npu_define_lib.define(
    "_allgather_in_tensor_shape(Tensor out, Tensor input, SymInt[] output_shape, \
        str tag, int[] ranks, int group_size) -> Tensor")

op_allgather_in_tensor_uneven = npu_define_lib.define(
    "allgather_in_tensor_uneven(Tensor input_tensor, SymInt send_count, SymInt[] recv_counts, \
    str tag, int[] rank_list, int group_size, SymInt[] recv_displacements) -> Tensor")


def allgather_in_tensor_npu(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        tag: str,
        ranks: List[int],
        group_size: int, ):
    pg = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    c10d.all_gather_into_tensor(output_tensor, input_tensor, group=pg, async_op=False)
    return output_tensor


def allgather_in_tensor_meta(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        tag: str,
        ranks: List[int],
        group_size: int, ):
    return output_tensor.new_empty(output_tensor.size())


@register_fx_node_ge_converter(torch.ops.npu_define.allgather_in_tensor.default)
def conveter_allgather_in_tensor(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        tag: str,
        rank_list,
        group_size: int,
        meta_outputs: Any = None,
):
    """allgather_in_tensor(Tensor out, Tensor input, str tag, int[] ranks, int group_size) -> Tensor"""
    group_name = get_group_name_and_record(tag, rank_list, group_size)
    res = ge.HcomAllGather(input_tensor, rank_size=group_size, group=group_name, fusion=0)
    return ge.Reshape(res, ge.Shape(output_tensor))


def _allgather_in_tensor_shape_npu(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_shape: List[int],
        tag: str,
        ranks: List[int],
        group_size: int, ):
    pg = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    c10d.all_gather_into_tensor(output_tensor, input_tensor, group=pg, async_op=False)
    return output_tensor


def _allgather_in_tensor_shape_meta(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_shape: List[int],
        tag: str,
        ranks: List[int],
        group_size: int, ):
    return torch.empty(output_shape, dtype=input_tensor.dtype, device="meta")


@register_fx_node_ge_converter(torch.ops.npu_define._allgather_in_tensor_shape.default)
def conveter_allgather_in_tensor_shape(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_shape: List[int],
        tag: str,
        rank_list,
        group_size: int,
        meta_outputs: Any = None,
):
    """allgather_in_tensor(Tensor out, Tensor input, str tag, int[] ranks, int group_size) -> Tensor"""
    group_name = get_group_name_and_record(tag, rank_list, group_size)
    res = ge.HcomAllGather(input_tensor, rank_size=group_size, group=group_name, fusion=0)
    return ge.Reshape(res, output_shape)


def npu_allgather_in_tensor_patch_dist(output_tensor, input_tensor, group=None, async_op=False):
    if not torch.distributed._functional_collectives._are_we_tracing():
        return torch.distributed.distributed_c10d.all_gather_into_tensor(output_tensor, input_tensor, group, async_op)
    if async_op:
        raise AssertionError(f'When you enable torch.compile or use the cache_compile feature, '
                       f'use the patch_for_hcom interface to ensure that collective communication functions '
                       f'are included in the graph. However, unlike the eager mode, the compile mode '
                       f'does not support the async_op = True parameter for collective communication APIs.')
    if group is None:
        group = c10d._world.default_pg
    ranklist = torch.distributed.get_process_group_ranks(group)
    tag = c10d._get_group_tag(group)
    size_ = len(ranklist)
    #allgather中的output_tensor输入只是使用它的size，并不会对tensor做改变，后续版本会变更为size输入
    out = torch.ops.npu_define._allgather_in_tensor_shape(output_tensor, input_tensor, \
        output_tensor.shape, tag, ranklist, size_)
    output_tensor.copy_(out)


def allgather_meta(
        output_tensor_list: List[Tensor],
        intput: Tensor,
        tag: str,
        ranks: List[int],
        group_size: int, ):
    return output_tensor_list


npu_define_lib.impl(op_allgather, allgather_meta, 'Meta')
npu_define_lib.impl(op_allgather_in_tensor, allgather_in_tensor_meta, 'Meta')
npu_define_lib.impl(op_allgather_in_tensor, allgather_in_tensor_npu, 'PrivateUse1')
npu_define_lib.impl(_op_allgather_in_tensor_shape, _allgather_in_tensor_shape_meta, 'Meta')
npu_define_lib.impl(_op_allgather_in_tensor_shape, _allgather_in_tensor_shape_npu, 'PrivateUse1')


def check_same_size(output_tensor_list):
    for _, t in enumerate(output_tensor_list):
        if t.size() != output_tensor_list[0].size():
            return False
    return True


def allgather_in_same_size(output_tensor_list, input_tensor, tag, ranks, group_size):
    output_shape_size = []
    for _, output_tensor in enumerate(output_tensor_list):
        output_shape_size.append(list(output_tensor.size()))

    ops_output_tensor = []
    recv_out_counts = []
    for _ in range(group_size):
        ops_output_tensor.append(input_tensor)
        recv_out_counts.append(input_tensor.shape[0])
    # 1、把input_tensor进行按照0维度进行拼接成一个tensor用于allgather_in_tensor输入
    ops_output_tensor_cat = torch.cat(ops_output_tensor, dim=0)
    out = torch.ops.npu_define.allgather_in_tensor(ops_output_tensor_cat, input_tensor, tag, ranks, group_size)
    # 2、返回值按照dim=0进行切分
    npu_output_tensor_list = list(torch.split(out, recv_out_counts, dim=0))
    # 3、reshape成用户size的tensor
    output_tensor_list_new = []
    for i, output_tensor in enumerate(npu_output_tensor_list):
        output_tensor_list_new.append(torch.reshape(output_tensor, output_shape_size[i]))
    return output_tensor_list_new


def allgather_in_different_size(output_tensor_list, input_tensor, tag, ranks, group_size):
    rank = torch.distributed.get_rank()
    npu_output_tensor_list = []
    for i, output_tensor in enumerate(output_tensor_list):
        ops_input_tensor = output_tensor
        if i == rank:
            ops_input_tensor = input_tensor
        out = torch.ops.npu_define.broadcast(ops_input_tensor, i, tag, ranks, group_size)
        npu_output_tensor_list.append(out.reshape(output_tensor.size()))
    return npu_output_tensor_list


@register_fx_node_ge_converter(torch.ops.npu_define.allgather_in_tensor_uneven.default)
def convert_allgather_in_tensor_uneven(
        input_tensor: Tensor,
        send_count: int,
        recv_counts: List[int],
        tag: str,
        rank_list: List,
        group_size: int,
        recv_displacements: List[int],
        meta_outputs: Any = None,
):
    send_count, recv_counts, recv_displacements = dtype_promote(send_count, recv_counts, recv_displacements,
                                                                target_dtype=DataType.DT_INT64)
    group_name = get_group_name_and_record(tag, rank_list, group_size)
    return ge.HcomAllGatherV(input_tensor, send_count=send_count, recv_counts=recv_counts,
                             recv_displacements=recv_displacements, group=group_name)


@register_decomposition(torch.ops.npu_define.allgather)
def allgather_decomposition(output_tensor_list: List[torch.Tensor],
                            input_tensor: Tensor,
                            tag: str,
                            ranks: List[int],
                            group_size: int,
                            ):
    if check_same_size(output_tensor_list):
        return allgather_in_same_size(output_tensor_list, input_tensor, tag, ranks, group_size)
    else:
        return allgather_in_different_size(output_tensor_list, input_tensor, tag, ranks, group_size)


def npu_all_gather_patch_dist(output_tensor_list, tensor, group=None, async_op=False):
    if not torch.distributed._functional_collectives._are_we_tracing():
        return torch.distributed.distributed_c10d.all_gather(output_tensor_list, tensor, group, async_op)
    if async_op:
        raise AssertionError(f'When you enable torch.compile or use the cache_compile feature, '
                       f'use the patch_for_hcom interface to ensure that collective communication functions '
                       f'are included in the graph. However, unlike the eager mode, the compile mode '
                       f'does not support the async_op = True parameter for collective communication APIs.')
    if group is None:
        group = c10d._world.default_pg
    ranklist = torch.distributed.get_process_group_ranks(group)
    tag = c10d._get_group_tag(group)
    # 判断2个shape是否相同
    # 判断返回output_tensor_list2个shape是否相同
    size_ = len(ranklist)
    if len(output_tensor_list) != size_:
        raise AssertionError(f'Tensor list input and rank size mismatch,\
        the len of list input is:{len(output_tensor_list)},but rank size is:{size_}.')
    npu_out_list = torch.ops.npu_define.allgather(output_tensor_list, tensor, tag, ranklist, size_)
    for i, _ in enumerate(output_tensor_list):
        output_tensor_list[i].copy_(npu_out_list[i])


def allgather_in_tensor_uneven_npu(
        input_tensor: torch.Tensor,
        send_count: int,
        recv_counts: List[int],
        tag: str,
        rank_list: List,
        group_size: int,
        recv_displacements: List[int]
):
    from torchair import ALL_GATHER_INTO_TENSOR_UNEVEN
    pg = c10d._find_or_create_pg_by_ranks_and_tag(tag, rank_list, group_size)
    out_size = list(input_tensor.size())
    out_size[0] = sum(recv_counts)
    out_tensor = input_tensor.new_empty(out_size)
    if ALL_GATHER_INTO_TENSOR_UNEVEN is None:
        raise AttributeError(f'torch_npu.distributed has no attribute: all_gather_into_tensor_uneven')
    ALL_GATHER_INTO_TENSOR_UNEVEN(out_tensor, input_tensor, recv_counts, group=pg, async_op=False)
    return out_tensor


def allgather_in_tensor_uneven_meta(
        input_tensor: torch.Tensor,
        send_count: int,
        recv_counts: List[int],
        tag: str,
        rank_list: List,
        group_size: int,
        recv_displacements: List[int]
):
    out_size = list(input_tensor.size())
    out_size[0] = sum(recv_counts)
    return input_tensor.new_empty(out_size)


npu_define_lib.impl(op_allgather_in_tensor_uneven, allgather_in_tensor_uneven_meta, 'Meta')
npu_define_lib.impl(op_allgather_in_tensor_uneven, allgather_in_tensor_uneven_npu, 'PrivateUse1')


def npu_allgather_into_tensor_uneven_patch_dist(output, input, output_split_sizes=None, group=None,
                                                async_op=False):
    if not torch.distributed._functional_collectives._are_we_tracing():
        from torchair import ALL_GATHER_INTO_TENSOR_UNEVEN
        if ALL_GATHER_INTO_TENSOR_UNEVEN is None:
            raise AttributeError(f'torch_npu.distributed has no attribute: all_gather_into_tensor_uneven')
        ALL_GATHER_INTO_TENSOR_UNEVEN(output, input, output_split_sizes, group, async_op)
        return
    if async_op:
        raise AssertionError(f'When you enable torch.compile or use the cache_compile feature, '
                             f'use the patch_for_hcom interface to ensure that collective communication functions '
                             f'are included in the graph. However, unlike the eager mode, the compile mode '
                             f'does not support the async_op = True parameter for collective communication APIs.')
    if group is None:
        group = c10d._world.default_pg
    rank_list = torch.distributed.get_process_group_ranks(group)
    tag = c10d._get_group_tag(group)
    group_size = len(rank_list)
    if output_split_sizes is None:
        output_split_sizes = [output.size(0) // group_size for _ in rank_list]
    if sum(output_split_sizes) != output.size(0):
        raise AssertionError(f'Split sizes sum does not match total dim 0 size')
    npu_output = torch.ops.npu_define.allgather_in_tensor_uneven(input, send_count=input.size(0),
                                                                 recv_counts=output_split_sizes, tag=tag,
                                                                 rank_list=rank_list, group_size=group_size,
                                                                 recv_displacements=[])
    output.copy_(npu_output.reshape(output.size()))
