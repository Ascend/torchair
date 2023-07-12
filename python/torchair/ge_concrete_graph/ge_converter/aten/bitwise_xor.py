import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided
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


@register_fx_node_ge_converter(torch.ops.aten.bitwise_xor.Tensor)
def conveter_aten_bitwise_xor_Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.bitwise_xor.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_xor.Scalar_Tensor)
def conveter_aten_bitwise_xor_Scalar_Tensor(
        self: Union[Number, Tensor],
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::bitwise_xor.Scalar_Tensor(Scalar self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.bitwise_xor.Scalar_Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_xor.Scalar)
def conveter_aten_bitwise_xor_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.bitwise_xor.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_xor.Tensor_out)
def conveter_aten_bitwise_xor_Tensor_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::bitwise_xor.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.bitwise_xor.Tensor_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_xor.Scalar_out)
def conveter_aten_bitwise_xor_Scalar_out(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::bitwise_xor.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.bitwise_xor.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_xor.Scalar_Tensor_out)
def conveter_aten_bitwise_xor_Scalar_Tensor_out(
        self: Union[Number, Tensor],
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::bitwise_xor.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.bitwise_xor.Scalar_Tensor_out ge converter is not implement!")


