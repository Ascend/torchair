import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
)


@register_fx_node_ge_converter(torch.ops.aten.gt.Tensor)
def conveter_aten_gt_Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::gt.Tensor(Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.gt.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.gt.Scalar)
def conveter_aten_gt_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::gt.Scalar(Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.gt.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.gt.Scalar_out)
def conveter_aten_gt_Scalar_out(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.gt.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.gt.Tensor_out)
def conveter_aten_gt_Tensor_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.gt.Tensor_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.gt.int)
def conveter_aten_gt_int(
        a: int,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::gt.int(int a, int b) -> bool """
    raise NotImplementedError("torch.ops.aten.gt.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.gt.float)
def conveter_aten_gt_float(
        a: float,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::gt.float(float a, float b) -> bool """
    raise NotImplementedError("torch.ops.aten.gt.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.gt.int_float)
def conveter_aten_gt_int_float(
        a: int,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::gt.int_float(int a, float b) -> bool """
    raise NotImplementedError("torch.ops.aten.gt.int_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.gt.float_int)
def conveter_aten_gt_float_int(
        a: float,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::gt.float_int(float a, int b) -> bool """
    raise NotImplementedError("torch.ops.aten.gt.float_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.gt.default)
def conveter_aten_gt_default(
        a: Union[Number, Tensor],
        b: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::gt(Scalar a, Scalar b) -> bool """
    raise NotImplementedError("torch.ops.aten.gt.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.gt.str)
def conveter_aten_gt_str(
        a: str,
        b: str,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::gt.str(str a, str b) -> bool """
    raise NotImplementedError("torch.ops.aten.gt.str ge converter is not implement!")


