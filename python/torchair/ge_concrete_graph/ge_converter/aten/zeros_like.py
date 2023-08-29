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
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.utils import dtype_promote
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support

    
@declare_supported(
    [
        Support(F32(8, 8), dtype=torch.int32),
        Support(F32(8, 8)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.zeros_like.default)
def conveter_aten_zeros_like_default(
    self: Tensor,
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    if dtype is not None:
        self = dtype_promote(self, target_dtype=dtype)
    if layout is not None and layout != torch.strided:
        raise NotImplementedError("torch.ops.aten.zeros_like.default ge_converter is only supported on denser tensor now!")
    return ge.ZerosLike(self)


@register_fx_node_ge_converter(torch.ops.aten.zeros_like.out)
def conveter_aten_zeros_like_out(
    self: Tensor,
    *,
    memory_format: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::zeros_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.zeros_like.out ge_converter is not implemented!")
