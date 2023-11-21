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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, DataType
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(3, 4, 5), 0.0),
    Support(F32(3, 4, 5), -1),
    Support(F16(3, 4, 5), -1),
    Support(I32(3, 4, 5), -1),
])
@register_fx_node_ge_converter(torch.ops.aten.clamp_min.default)
def conveter_aten_clamp_min_default(
    self: Tensor, min: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_min(Tensor self, Scalar min) -> Tensor"""
    if self.dtype == DataType.DT_INT32 or self.dtype == DataType.DT_INT64:
        max_value = torch.iinfo(torch.int32).max
    elif self.dtype == DataType.DT_FLOAT:
        max_value = torch.finfo(torch.float32).max
    else:
        max_value = torch.finfo(torch.float16).max
    min_value = dtype_promote(min, target_dtype=self.dtype)
    max_value = dtype_promote(max_value, target_dtype=self.dtype)
    return ge.ClipByValueV2(self, min_value, max_value)


@register_fx_node_ge_converter(torch.ops.aten.clamp_min.Tensor)
def conveter_aten_clamp_min_Tensor(self: Tensor, min: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::clamp_min.Tensor(Tensor self, Tensor min) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.clamp_min.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_min.out)
def conveter_aten_clamp_min_out(
    self: Tensor,
    min: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp_min.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_min.Tensor_out)
def conveter_aten_clamp_min_Tensor_out(
    self: Tensor, min: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_min.Tensor_out(Tensor self, Tensor min, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp_min.Tensor_out ge_converter is not implemented!")
