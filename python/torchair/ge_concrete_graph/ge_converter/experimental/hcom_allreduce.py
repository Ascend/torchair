from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable
import torch
from torch.library import Library
from torchair.ge_concrete_graph.ge_graph import Tensor
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter


_lib = Library("npu_define", "DEF")
op_name = _lib.define(
    "allreduce(Tensor input, str reduce_type, int[] ranks, float? timeout=None) -> Tensor")


def allreduce_cpu(inputs, reduce_type, ranks, timeout=None):
    if custom_all_reduce is None:
        torch_all_reduce(inputs)
    else:
        custom_all_reduce(inputs)
    return inputs


def allreduce_npu(inputs, reduce_type, ranks, timeout=None):
    if custom_all_reduce is None:
        torch_all_reduce(inputs)
    else:
        custom_all_reduce(inputs)
    return inputs


def allreduce_meta(inputs, reduce_type, ranks, timeout=None):
    return inputs


_lib.impl(op_name, allreduce_cpu, 'CPU')
_lib.impl(op_name, allreduce_meta, 'Meta')
_lib.impl(op_name, allreduce_npu, 'PrivateUse1')


def npu_all_reduce(tensor, op="sum", group=None, async_op=False):
    tensor.copy_(torch.ops.npu_define.allreduce(tensor, "sum", []))


torch_all_reduce = torch.distributed.all_reduce
torch.distributed.all_reduce = npu_all_reduce
custom_all_reduce = None


def get_npu_all_reduce():
    return npu_all_reduce


def backup_custom_all_reduce(func: None):
    if func is not None:
        custom_all_reduce = func


@register_fx_node_ge_converter(torch.ops.npu_define.allreduce.default)
def conveter_allreduce(
        self: Tensor,
        reduce_type,
        ranklist,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    from torch.distributed.distributed_c10d import _world
    rank = torch.distributed.get_rank()
    ranklist = torch.distributed.get_process_group_ranks(_world.default_pg)
    y = ge.HcomAllReduce(self, reduction=reduce_type, group="hccl_world_group", fusion=0)
    y._node.attr["ranklist"].list.i.extend(ranklist)
    try:
        hcom_info = _world.default_pg._get_backend(torch.device("npu")).get_hccl_comm(rank)
    except:
        pass
    else:
        y._node.attr["comm"].i = hcom_info
    return y
