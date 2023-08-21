from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, torch_type_to_ge_type
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten._to_copy.default)
def conveter_aten__to_copy_default(
    self: Tensor,
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    non_blocking: bool = False,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor"""
    # layout, pin_memory, device and non_blocking have no effect on constructing graph.
    if memory_format is not None and memory_format is not torch.contiguous_format:
        raise NotImplementedError(
            "torch.ops.aten._to_copy.default have some unprocessed parameters or cases, "
            "memory_format = {}, torch.contiguous_format = {}".format(memory_format, torch.contiguous_format))

    if dtype is None:
        return ge.Identity(self)
    else:
        return ge.Cast(self, dst_type=torch_type_to_ge_type(dtype))


@register_fx_node_ge_converter(torch.ops.aten._to_copy.out)
def conveter_aten__to_copy_out(
    self: Tensor,
    *,
    non_blocking: bool = False,
    memory_format: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_to_copy.out(Tensor self, *, bool non_blocking=False, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._to_copy.out ge_converter is not implemented!")
