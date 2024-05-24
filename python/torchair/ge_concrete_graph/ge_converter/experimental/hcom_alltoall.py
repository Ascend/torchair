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


op_all_to_all_single = npu_define_lib.define(
    "all_to_all_single(Tensor input, SymInt[]? output_split_sizes, \
     SymInt[]? input_split_sizes, str tag, int[] ranks, int group_size) -> Tensor")

op_all_to_all_single_npu = npu_define_lib.define(
    "all_to_all_single_npu(Tensor input, SymInt[] send_counts, SymInt[] send_displacements, \
     SymInt[] recv_counts, SymInt[] recv_displacements, str tag, int[] ranks, int group_size) -> Tensor")


@register_decomposition(torch.ops.npu_define.all_to_all_single)
def all_to_all_single_decomposition(
    input_tensor: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
    tag: str,
    ranks: List[int],
    group_size: int,
):
    if input_split_sizes is None and output_split_sizes is None:
        input_split_sizes = []
        output_split_sizes = []
        spilt_size = input_tensor.numel() // group_size
        for i in range(group_size):
            input_split_sizes.append(spilt_size)
            output_split_sizes.append(spilt_size)

    send_counts = []
    recv_counts = []
    send_displacements = [0]
    recv_displacements = [0]
    if len(input_split_sizes) != len(output_split_sizes):
        raise AssertionError
    for i, split_size in enumerate(input_split_sizes):
        send_counts.append(split_size)
        if i > 0:
            send_displacements.append(send_displacements[i - 1] + send_counts[i - 1])

    for i, split_size in enumerate(output_split_sizes):
        recv_counts.append(split_size)
        if i > 0:
            recv_displacements.append(recv_displacements[i - 1] + recv_counts[i - 1])

    return torch.ops.npu_define.all_to_all_single_npu(input_tensor,
        send_counts, send_displacements, recv_counts, recv_displacements, tag, ranks, group_size)


def npu_all_to_all_single(
    input_tensor: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
    tag: str,
    ranks: List[int],
    group_size: int,
):
    pg = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)

    if output_split_sizes is not None:
        if input_tensor.dim() == 0: 
            raise AssertionError(input_tensor.dim())
        out_size = list(input_tensor.size())
        out_size[0] = sum(output_split_sizes)
        out_tensor = input_tensor.new_empty(out_size)
    else:
        out_tensor = input_tensor.new_empty(input_tensor.size())

    work = c10d.all_to_all_single(
        out_tensor, input_tensor, output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes, group=pg, async_op=False
    )
    return out_tensor


def npu_all_to_all_single_meta(
    input_tensor,
    output_split_sizes,
    input_split_sizes,
    tag,
    ranks,
    group_size
):
    if output_split_sizes is None:
        return input_tensor.new_empty(input_tensor.size())
    else:
        # TO DO check 非负,且不能影响符号化 2.3有接口，2.1暂未实现 后期对接官方算子 临时方案先不实现
        out_size = list(input_tensor.size())
        out_size[0] = sum(output_split_sizes)
        return input_tensor.new_empty(out_size)


npu_define_lib.impl(op_all_to_all_single, npu_all_to_all_single, 'CPU')
npu_define_lib.impl(op_all_to_all_single, npu_all_to_all_single, 'PrivateUse1')
npu_define_lib.impl(op_all_to_all_single, npu_all_to_all_single_meta, 'Meta')


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


@register_fx_node_ge_converter(torch.ops.npu_define.all_to_all_single_npu.default)
def convert_all_to_all_single_npu(
    input_tensor: Tensor,
    send_counts: Union[Tensor, List[int]],
    send_displacements: Union[Tensor, List[int]],
    recv_counts: Union[Tensor, List[int]],
    recv_displacements: Union[Tensor, List[int]],
    tag: str,
    ranklist: List,
    group_size: int,
    out: Tensor = None,
    meta_outputs: Any = None,
):
    """NB: npu_define::all_to_all_single_npu(Tensor input, SymInt[] send_counts, SymInt[] send_displacements, \
        SymInt[] recv_counts, SymInt[] recv_displacements, str tag, int[] ranks, int group_size) -> Tensor"""
    rank = torch.distributed.get_rank()
    pg = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranklist, group_size)
    device = torch.distributed.distributed_c10d._get_pg_default_device(pg)
    send_counts, send_displacements, recv_counts, recv_displacements = dtype_promote(send_counts, 
        send_displacements, recv_counts, recv_displacements, target_dtype=DataType.DT_INT64)
    if device.type == "cpu":
        y = ge.HcomAllToAllV(send_data=input_tensor, send_counts=send_counts, send_displacements=send_displacements,
                             recv_counts=recv_counts, recv_displacements=recv_displacements, group=tag)
    elif device.type == "npu":
        hcom_name = pg._get_backend(device).get_hccl_comm_name(rank)
        y = ge.HcomAllToAllV(send_data=input_tensor, send_counts=send_counts, send_displacements=send_displacements,
                             recv_counts=recv_counts, recv_displacements=recv_displacements, group=hcom_name)
    else:
        raise ValueError("The initialized aggregate communication backend is not a CPU or NPU.")
    y._node.attr["ranklist"].list.i.extend(ranklist)
    return y


def npu_all_to_all_single_patch_dist(
    output_tensor,
    input_tensor,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    async_op=False,
):
    c10d._check_single_tensor(output_tensor, "output_tensor")
    c10d._check_single_tensor(input_tensor, "input_tensor")
    c10d._ensure_all_tensors_same_dtype(output_tensor, input_tensor)

    # TO DO 由于目前dynamo不支持 is_complex，暂不支持复数tensor入图(原生接口中做了转化)
    if group is None:
        group = c10d._world.default_pg
    ranklist = torch.distributed.get_process_group_ranks(group)
    tag = c10d._get_group_tag(group)
    npu_out = torch.ops.npu_define.all_to_all_single(input_tensor, output_split_sizes, input_split_sizes,
                                                     tag, ranklist, len(ranklist))
    output_tensor.copy_(npu_out)


torch.distributed.all_to_all_single = npu_all_to_all_single_patch_dist

# 谨记 这个是非原地语义 output入参是为了表达 output tensor中的shape信息，不能使用其地址
op_all_to_all = npu_define_lib.define(
    "all_to_all(Tensor[] input, Tensor[] output, str tag, int[] ranks, int group_size) -> Tensor[]")


def npu_all_to_all(
        input_tensor_list: List[torch.Tensor],
        output_tensor_list: List[torch.Tensor],
        tag: str,
        ranks: List[int],
        group_size: int,
):
    pg = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)

    npu_output_tensor_list = []
    for i, output_tensor in enumerate(output_tensor_list):
        npu_output_tensor_list.append(torch.empty_like(output_tensor))

    work = c10d.all_to_all(npu_output_tensor_list, input_tensor_list, group=pg, async_op=False)

    return npu_output_tensor_list


def npu_all_to_all_meta(
        input_tensor_list: List[torch.Tensor],
        output_tensor_list: List[torch.Tensor],
        tag: str,
        ranks: List[int],
        group_size: int,
):
    npu_output_tensor_list = []
    for i, output_tensor in enumerate(output_tensor_list):
        npu_output_tensor_list.append(torch.empty_like(output_tensor))
    return npu_output_tensor_list


npu_define_lib.impl(op_all_to_all, npu_all_to_all, 'PrivateUse1')
npu_define_lib.impl(op_all_to_all, npu_all_to_all_meta, 'Meta')


@register_decomposition(torch.ops.npu_define.all_to_all)
def all_to_all_decomposition(
    input_tensor_list: List[torch.Tensor],
    output_tensor_list: List[torch.Tensor],
    tag: str,
    ranks: List[int],
    group_size: int,
):
    output_shape_size = []
    send_counts = []
    send_displacements = [0]
    recv_counts = []
    recv_displacements = [0]
    for i, input_tensor in enumerate(input_tensor_list):
        send_counts.append(input_tensor.numel())
        if i > 0:
            send_displacements.append(send_displacements[i - 1] + send_counts[i - 1])

    for i, output_tensor in enumerate(output_tensor_list):
        output_shape_size.append(list(output_tensor.size()))
        recv_counts.append(output_tensor.numel())
        if i > 0:
            recv_displacements.append(recv_displacements[i - 1] + recv_counts[i - 1])

    for i, input_tensor in enumerate(input_tensor_list):
        input_tensor_list[i] = input_tensor.flatten()
    input_tensor = torch.cat(input_tensor_list, dim=0)

    output_tensor = torch.ops.npu_define.all_to_all_single_npu(input_tensor,
        send_counts, send_displacements, recv_counts, recv_displacements, tag, ranks, group_size)

    npu_output_tensor_list = list(torch.split(output_tensor, recv_counts, dim=0))

    for i, output_tensor in enumerate(npu_output_tensor_list):
        npu_output_tensor_list[i] = torch.reshape(output_tensor, output_shape_size[i])
    return npu_output_tensor_list


def npu_all_to_all_patch_dist(
        output_tensor_list: List[torch.Tensor],
        input_tensor_list: List[torch.Tensor],
        group=None,
        async_op=False,
):
    if len(input_tensor_list) != len(output_tensor_list):
        raise AssertionError
    if group is None:
        group = c10d._world.default_pg
    ranklist = torch.distributed.get_process_group_ranks(group)
    tag = c10d._get_group_tag(group)
    c10d._check_tensor_list(output_tensor_list, "output_tensor_list")
    c10d._check_tensor_list(input_tensor_list, "input_tensor_list")
    c10d._ensure_all_tensors_same_dtype(output_tensor_list, input_tensor_list)

    npu_out_list = torch.ops.npu_define.all_to_all(input_tensor_list, output_tensor_list, tag, ranklist, len(ranklist))
    if len(npu_out_list) != len(output_tensor_list):
        raise AssertionError(f'The expect npu_out_list len {len(output_tensor_list)}, but got {len(npu_out_list)}.')

    for i, _ in enumerate(output_tensor_list):
        output_tensor_list[i].copy_(npu_out_list[i])


torch.distributed.all_to_all = npu_all_to_all_patch_dist
