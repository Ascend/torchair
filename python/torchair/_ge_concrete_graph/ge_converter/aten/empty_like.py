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
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.aten.empty_like.default)
def conveter_aten_empty_like_default(
    self: Tensor,
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    
    if dtype is None:
        dtype = self.dtype

    if layout is not None and (layout != torch.strided):
        raise RuntimeError("torch.ops.aten.empty_like.default is only supported on dense tensor now.")
    
    if memory_format is not None and memory_format != torch.contiguous_format \
            and memory_format != torch.preserve_format:
        raise RuntimeError("torch.ops.aten.empty_like.default is only supported "
                "contiguous_format and preserve_format now.")
    # There is a bug with the op Empty when dynamic=True and dtype=int8.
    # So replace Empty with Fill.
    return ge.Fill(ge.Shape(self), ge.Cast(0., dst_type=meta_outputs.dtype))


@register_fx_node_ge_converter(torch.ops.aten.empty_like.out)
def conveter_aten_empty_like_out(
    self: Tensor,
    *,
    memory_format: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::empty_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.empty_like.out ge_converter is "
                       "redundant before pytorch 2.1.0, might be supported in future version.")
