from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable
import torch
import torch.distributed.distributed_c10d as c10d
from torchair.ge._ge_graph import Tensor, DataType, dont_prune_me
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.hcom_utils import get_group_name_and_record
from .hcom_allreduce import npu_define_lib


op_all_to_all_single_npu = npu_define_lib.define(
    "all_to_all_single_npu(Tensor input, SymInt[] send_counts, SymInt[] send_displacements, \
     SymInt[] recv_counts, SymInt[] recv_displacements, str tag, int[] ranks, int group_size) -> Tensor")


def _all_to_all_single(
    input_tensor: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
    tag: str,
    ranks: List[int],
    group_size: int,
):
    if input_split_sizes is None and output_split_sizes is None:
        input_split_sizes = [input_tensor.numel() // group_size] * group_size
        output_split_sizes = [input_tensor.numel() // group_size] * group_size

    if len(input_split_sizes) != len(output_split_sizes):
        raise AssertionError

    out_size = list(input_tensor.size())
    if len(out_size) != 1:
        raise AssertionError(f"Expected the input tensor used in all_to_all_single to have a dimensionality of 1,"
                             f"but got {len(out_size)}")
    send_displacements = [sum(input_split_sizes[:i]) for i in range(len(input_split_sizes))]
    recv_displacements = [sum(output_split_sizes[:i]) for i in range(len(output_split_sizes))]
    return torch.ops.npu_define.all_to_all_single_npu(input_tensor,
        input_split_sizes, send_displacements, output_split_sizes, recv_displacements, tag, ranks, group_size)


def npu_all_to_all_single_npu(
    input_tensor: Tensor,
    send_counts: List[int],
    send_displacements: List[int],
    recv_counts: List[int],
    recv_displacements: List[int],
    tag: str,
    ranklist: List,
    group_size: int,
):
    pg = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranklist, group_size)

    if recv_counts is not None:
        if input_tensor.dim() == 0:
            raise AssertionError(input_tensor.dim())
        out_size = list(input_tensor.size())
        out_size[0] = sum(recv_counts)
        out_tensor = input_tensor.new_empty(out_size)
    else:
        out_tensor = input_tensor.new_empty(input_tensor.size())

    work = c10d.all_to_all_single(out_tensor, input_tensor, output_split_sizes=recv_counts,
                                  input_split_sizes=send_counts, group=pg, async_op=False)
    return out_tensor


def npu_all_to_all_single_npu_meta(
    input_tensor: Tensor,
    send_counts: List[int],
    send_displacements: List[int],
    recv_counts: List[int],
    recv_displacements: List[int],
    tag: str,
    ranklist: List,
    group_size: int,
):
    out_size = list(input_tensor.size())
    out_size[0] = sum(recv_counts)
    return input_tensor.new_empty(out_size)


npu_define_lib.impl(op_all_to_all_single_npu, npu_all_to_all_single_npu_meta, 'Meta')
npu_define_lib.impl(op_all_to_all_single_npu, npu_all_to_all_single_npu, 'PrivateUse1')


@register_fx_node_ge_converter(torch.ops.npu_define.all_to_all_single_npu.default)
def convert_all_to_all_single_npu(
        input_tensor: Tensor,
        send_counts: Union[Tensor, List[int]],
        send_displacements: Union[Tensor, List[int]],
        recv_counts: Union[Tensor, List[int]],
        recv_displacements: Union[Tensor, List[int]],
        tag: str,
        rank_list: List,
        group_size: int,
        out: Tensor = None,
        meta_outputs: Any = None,
):
    """NB: npu_define::all_to_all_single_npu(Tensor input, SymInt[] send_counts, SymInt[] send_displacements, \
        SymInt[] recv_counts, SymInt[] recv_displacements, str tag, int[] ranks, int group_size) -> Tensor"""
    group_name = get_group_name_and_record(tag, rank_list, group_size)

    try:
        from torch_npu.npu.utils import _is_gte_cann_version
        is_supported_version = _is_gte_cann_version("8.2.RC1", module="CANN")
    except (ImportError, RuntimeError):
        is_supported_version = False
    check_all_to_all_supported = isinstance(send_counts, List) and isinstance(recv_counts, List) and len(
        set(send_counts + recv_counts)) == 1
    
    # TO DO: fix me
    is_supported_version = False

    if is_supported_version and check_all_to_all_supported:
        return ge.HcomAllToAll(input_tensor, group=group_name)

    send_counts, send_displacements, recv_counts, recv_displacements = dtype_promote(send_counts,
                                                                                     send_displacements,
                                                                                     recv_counts,
                                                                                     recv_displacements,
                                                                                     target_dtype=DataType.DT_INT64)
    op = ge.HcomAllToAllV(send_data=input_tensor, send_counts=send_counts, send_displacements=send_displacements,
                          recv_counts=recv_counts, recv_displacements=recv_displacements, group=group_name)
    dont_prune_me(op)
    return op


def npu_all_to_all_single_patch_dist(
    output_tensor,
    input_tensor,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    async_op=False,
):
    if not torch.distributed._functional_collectives._are_we_tracing():
        return torch.distributed.distributed_c10d.all_to_all_single(output_tensor,
                                                                    input_tensor,
                                                                    output_split_sizes,
                                                                    input_split_sizes,
                                                                    group,
                                                                    async_op)
    if async_op:
        raise AssertionError(f'When you enable torch.compile or use the cache_compile feature, '
                       f'use the patch_for_hcom interface to ensure that collective communication functions '
                       f'are included in the graph. However, unlike the eager mode, the compile mode '
                       f'does not support the async_op = True parameter for collective communication APIs.')

    c10d._check_single_tensor(output_tensor, "output_tensor")
    c10d._check_single_tensor(input_tensor, "input_tensor")
    c10d._ensure_all_tensors_same_dtype(output_tensor, input_tensor)

    # TO DO 由于目前dynamo不支持 is_complex，暂不支持复数tensor入图(原生接口中做了转化)
    if group is None:
        group = c10d._world.default_pg
    ranklist = torch.distributed.get_process_group_ranks(group)
    tag = c10d._get_group_tag(group)
    npu_out = _all_to_all_single(input_tensor, output_split_sizes, input_split_sizes,
                                 tag, ranklist, len(ranklist))
    output_tensor.copy_(npu_out)


def all_to_all_decomposition(
    input_tensor_list: List[torch.Tensor],
    output_tensor_list: List[torch.Tensor],
    tag: str,
    ranks: List[int],
    group_size: int,
):
    send_counts = [tensor.numel() for tensor in input_tensor_list]
    recv_counts = [tensor.numel() for tensor in output_tensor_list]
    send_displacements = [sum(send_counts[:i]) for i in range(len(send_counts))]
    recv_displacements = [sum(recv_counts[:i]) for i in range(len(recv_counts))]
    output_shape_size = [list(tensor.size()) for tensor in output_tensor_list]

    input_tensor_list = [tensor.flatten() for tensor in input_tensor_list]
    input_tensor = torch.cat(input_tensor_list, dim=0)

    output_tensor = torch.ops.npu_define.all_to_all_single_npu(input_tensor,
        send_counts, send_displacements, recv_counts, recv_displacements, tag, ranks, group_size)

    npu_output_tensor_list = [
        torch.reshape(tensor, output_shape_size[i])
        for i, tensor in enumerate(torch.split(output_tensor, recv_counts, dim=0))
    ]
    return npu_output_tensor_list


def npu_all_to_all_patch_dist(
        output_tensor_list: List[torch.Tensor],
        input_tensor_list: List[torch.Tensor],
        group=None,
        async_op=False,
):
    if not torch.distributed._functional_collectives._are_we_tracing():
        return torch.distributed.distributed_c10d.all_to_all(output_tensor_list, input_tensor_list, group, async_op)
    if async_op:
        raise AssertionError(f'When you enable torch.compile or use the cache_compile feature, '
                       f'use the patch_for_hcom interface to ensure that collective communication functions '
                       f'are included in the graph. However, unlike the eager mode, the compile mode '
                       f'does not support the async_op = True parameter for collective communication APIs.')

    if len(input_tensor_list) != len(output_tensor_list):
        raise AssertionError
    if group is None:
        group = c10d._world.default_pg
    ranklist = torch.distributed.get_process_group_ranks(group)
    tag = c10d._get_group_tag(group)
    c10d._check_tensor_list(output_tensor_list, "output_tensor_list")
    c10d._check_tensor_list(input_tensor_list, "input_tensor_list")
    c10d._ensure_all_tensors_same_dtype(output_tensor_list, input_tensor_list)

    npu_out_list = all_to_all_decomposition(input_tensor_list, output_tensor_list, tag, ranklist, len(ranklist))
    if len(npu_out_list) != len(output_tensor_list):
        raise AssertionError(f'The expect npu_out_list len {len(output_tensor_list)}, but got {len(npu_out_list)}.')

    for i, _ in enumerate(output_tensor_list):
        output_tensor_list[i].copy_(npu_out_list[i])
