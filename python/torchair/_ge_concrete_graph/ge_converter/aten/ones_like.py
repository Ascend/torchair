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
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter, \
    torch_type_to_ge_type
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2,2)),
    Support(F32(2, 2), dtype=torch.int),
])
@register_fx_node_ge_converter(torch.ops.aten.ones_like.default)
def conveter_aten_ones_like_default(
    self: Tensor,
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    if dtype is None:
        dtype = self.dtype
    else:
        dtype = torch_type_to_ge_type(dtype)
    
    if layout and layout != torch.strided:
        raise NotImplementedError("torch.ops.aten.ones_like.out ge_converter is only supportded on dense tensor!")
    self = dtype_promote(self, target_dtype=dtype)
    return ge.OnesLike(self)

@register_fx_node_ge_converter(torch.ops.aten.ones_like.out)
def conveter_aten_ones_like_out(
    self: Tensor,
    *,
    memory_format: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::ones_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.ones_like.out ge_converter is not supported!")
