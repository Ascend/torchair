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
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, DataType
from torchair.ge_concrete_graph.utils import dtype_promote
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F32(10,), None, 2),
        Support(F32(10,), 1),
        Support(F32(10,), 1, 3)
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.clamp.default)
def conveter_aten_clamp_default(
    self: Tensor,
    min: Optional[Union[Number, Tensor]] = None,
    max: Optional[Union[Number, Tensor]] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor"""
    if not min and not max:
        raise NotImplementedError("torch.clamp: At least one of 'min' or 'max' must not be None")
    if not min:
        if self.dtype == DataType.DT_INT32 or self.dtype == DataType.DT_INT64:
            min = torch.iinfo(torch.int32).min
        elif self.dtype == DataType.DT_FLOAT:
            min = torch.finfo(torch.float32).min
        else:
            min = torch.finfo(torch.float16).min
    if not max:
        if self.dtype == DataType.DT_INT32 or self.dtype == DataType.DT_INT64:
            max = torch.iinfo(torch.int32).max
        elif self.dtype == DataType.DT_FLOAT:
            max = torch.finfo(torch.float32).max
        else:
            max = torch.finfo(torch.float16).max
    min = dtype_promote(min, target_dtype=self.dtype)
    max = dtype_promote(max, target_dtype=self.dtype)
    return ge.ClipByValue(self, min, max)


@register_fx_node_ge_converter(torch.ops.aten.clamp.Tensor)
def conveter_aten_clamp_Tensor(
    self: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.clamp.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clamp.out)
def conveter_aten_clamp_out(
    self: Tensor,
    min: Optional[Union[Number, Tensor]] = None,
    max: Optional[Union[Number, Tensor]] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clamp.Tensor_out)
def conveter_aten_clamp_Tensor_out(
    self: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::clamp.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp.Tensor_out ge_converter is not implemented!")
