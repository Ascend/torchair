from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable
import torch
from torch.library import Library, impl
from torchair.ge_concrete_graph.ge_graph import get_default_ge_graph, next_unique_name, auto_convert_to_tensor
from torchair.ge_concrete_graph.ge_graph import Tensor
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.core.utils import logger

_lib = Library("npu_define", "DEF")
op_name = _lib.define(
    "allreduce(Tensor input, str reduce_type, int[] ranks, float? timeout=None) -> Tensor")


def allreduce_cpu(inputs, reduce_type, ranks, timeout=None):
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
    # Work with PyTorch, rank list has no real meaning
    # Work without PyTorch, like inference without PyTorch, it will be modiefed later, default 0-3
    default_ranklist = [0, 1, 2, 3]
    tensor.copy_(torch.ops.npu_define.allreduce(tensor, "sum", default_ranklist))


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
    rank = torch.distributed.get_rank()
    from torch.distributed.distributed_c10d import _world
    hcom_info = _world.default_pg._get_backend(
        torch.device("npu")).get_hccl_comm(rank)
    y = ge.HcomAllReduce(self, reduction=reduce_type,
                         group="hccl_world_group", fusion=0)
    y._node.attr["comm"].i = hcom_info
    y._node.attr["ranklist"].list.i.extend(ranklist)
    return y
