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


@register_fx_node_ge_converter(torch.ops.aten.max.other)
def conveter_aten_max_other(
        self: Tensor,
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::max.other(Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.max.other ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.max.default)
def conveter_aten_max_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::max(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.max.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.max.dim)
def conveter_aten_max_dim(
        self: Tensor,
        dim: int,
        keepdim: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices) """
    raise NotImplementedError("torch.ops.aten.max.dim ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.max.dim_max)
def conveter_aten_max_dim_max(
        self: Tensor,
        dim: int,
        keepdim: bool = False,
        *,
        max: Tensor = None,
        max_values: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices) """
    raise NotImplementedError("torch.ops.aten.max.dim_max ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.max.names_dim)
def conveter_aten_max_names_dim(
        self: Tensor,
        dim: str,
        keepdim: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::max.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices) """
    raise NotImplementedError("torch.ops.aten.max.names_dim ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.max.names_dim_max)
def conveter_aten_max_names_dim_max(
        self: Tensor,
        dim: str,
        keepdim: bool = False,
        *,
        max: Tensor = None,
        max_values: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::max.names_dim_max(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices) """
    raise NotImplementedError("torch.ops.aten.max.names_dim_max ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.max.unary_out)
def conveter_aten_max_unary_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::max.unary_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.max.unary_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.max.out)
def conveter_aten_max_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.max.out ge converter is not implement!")


