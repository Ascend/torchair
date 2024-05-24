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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F32(96, 384, 14, 14), 0.0, 6.0),
        Support(F16(96, 384, 14, 14), 0.0, 6.0),
        Support(F32(2, 1280, 7, 7), 0.0, 6.0),
        Support(F32(2, 1280, 7, 7), 2.0, -2.0),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.hardtanh.default)
def conveter_aten_hardtanh_default(
    self: Tensor,
    min_val: Union[Number, Tensor] = -1,
    max_val: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor"""
    if min_val >= max_val:
        return ge.Fill(ge.Shape(self), ge.Cast(max_val, dst_type=self.dtype))
    return ge.ClipByValue(self, min_val, max_val)


@register_fx_node_ge_converter(torch.ops.aten.hardtanh.out)
def conveter_aten_hardtanh_out(
    self: Tensor,
    min_val: Union[Number, Tensor] = -1,
    max_val: Union[Number, Tensor] = 1,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.hardtanh.out is redundant before pytorch 2.1.0,"
                       "might be supported in future version.")
