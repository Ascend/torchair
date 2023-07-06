import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
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


@register_fx_node_ge_converter(torch.ops.aten.scatter.value)
def conveter_aten_scatter_value(
        self: Tensor,
        dim: int,
        index: Tensor,
        value: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor """
    raise NotImplementedError("torch.ops.aten.scatter.value ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.src)
def conveter_aten_scatter_src(
        self: Tensor,
        dim: int,
        index: Tensor,
        src: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor """
    raise NotImplementedError("torch.ops.aten.scatter.src ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.reduce)
def conveter_aten_scatter_reduce(
        self: Tensor,
        dim: int,
        index: Tensor,
        src: Tensor,
        *,
        reduce: str,
        meta_outputs: Any = None):
    """ NB: aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor """
    raise NotImplementedError("torch.ops.aten.scatter.reduce ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.value_reduce)
def conveter_aten_scatter_value_reduce(
        self: Tensor,
        dim: int,
        index: Tensor,
        value: Union[Number, Tensor],
        *,
        reduce: str,
        meta_outputs: Any = None):
    """ NB: aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor """
    raise NotImplementedError("torch.ops.aten.scatter.value_reduce ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.src_out)
def conveter_aten_scatter_src_out(
        self: Tensor,
        dim: int,
        index: Tensor,
        src: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::scatter.src_out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.scatter.src_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.value_out)
def conveter_aten_scatter_value_out(
        self: Tensor,
        dim: int,
        index: Tensor,
        value: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::scatter.value_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.scatter.value_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.reduce_out)
def conveter_aten_scatter_reduce_out(
        self: Tensor,
        dim: int,
        index: Tensor,
        src: Tensor,
        *,
        reduce: str,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::scatter.reduce_out(Tensor self, int dim, Tensor index, Tensor src, *, str reduce, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.scatter.reduce_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.value_reduce_out)
def conveter_aten_scatter_value_reduce_out(
        self: Tensor,
        dim: int,
        index: Tensor,
        value: Union[Number, Tensor],
        *,
        reduce: str,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::scatter.value_reduce_out(Tensor self, int dim, Tensor index, Scalar value, *, str reduce, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.scatter.value_reduce_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.dimname_src)
def conveter_aten_scatter_dimname_src(
        self: Tensor,
        dim: str,
        index: Tensor,
        src: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::scatter.dimname_src(Tensor self, str dim, Tensor index, Tensor src) -> Tensor """
    raise NotImplementedError("torch.ops.aten.scatter.dimname_src ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.dimname_value)
def conveter_aten_scatter_dimname_value(
        self: Tensor,
        dim: str,
        index: Tensor,
        value: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::scatter.dimname_value(Tensor self, str dim, Tensor index, Scalar value) -> Tensor """
    raise NotImplementedError("torch.ops.aten.scatter.dimname_value ge converter is not implement!")


