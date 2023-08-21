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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.aten.gt.Tensor)
def conveter_aten_gt_Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::gt.Tensor(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.gt.Tensor ge_converter is not implemented!")


@declare_supported([
    Support(F32(1024, 1024), 0),
])
@register_fx_node_ge_converter(torch.ops.aten.gt.Scalar)
def conveter_aten_gt_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::gt.Scalar(Tensor self, Scalar other) -> Tensor"""
    return ge.Greater(self, other)


@register_fx_node_ge_converter(torch.ops.aten.gt.Scalar_out)
def conveter_aten_gt_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.gt.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gt.Tensor_out)
def conveter_aten_gt_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.gt.Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gt.int)
def conveter_aten_gt_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::gt.int(int a, int b) -> bool"""
    raise NotImplementedError("torch.ops.aten.gt.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gt.float)
def conveter_aten_gt_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::gt.float(float a, float b) -> bool"""
    raise NotImplementedError("torch.ops.aten.gt.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gt.int_float)
def conveter_aten_gt_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::gt.int_float(int a, float b) -> bool"""
    raise NotImplementedError("torch.ops.aten.gt.int_float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gt.float_int)
def conveter_aten_gt_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::gt.float_int(float a, int b) -> bool"""
    raise NotImplementedError("torch.ops.aten.gt.float_int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gt.default)
def conveter_aten_gt_default(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::gt(Scalar a, Scalar b) -> bool"""
    raise NotImplementedError("torch.ops.aten.gt.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gt.str)
def conveter_aten_gt_str(a: str, b: str, meta_outputs: TensorSpec = None):
    """NB: aten::gt.str(str a, str b) -> bool"""
    raise NotImplementedError("torch.ops.aten.gt.str ge_converter is not implemented!")
