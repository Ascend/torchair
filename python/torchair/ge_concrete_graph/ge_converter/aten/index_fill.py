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


@register_fx_node_ge_converter(torch.ops.aten.index_fill.int_Tensor)
def conveter_aten_index_fill_int_Tensor(
        self: Tensor,
        dim: int,
        index: Tensor,
        value: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor """
    raise NotImplementedError("torch.ops.aten.index_fill.int_Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill.int_Scalar)
def conveter_aten_index_fill_int_Scalar(
        self: Tensor,
        dim: int,
        index: Tensor,
        value: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor """
    raise NotImplementedError("torch.ops.aten.index_fill.int_Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill.Dimname_Scalar)
def conveter_aten_index_fill_Dimname_Scalar(
        self: Tensor,
        dim: str,
        index: Tensor,
        value: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index_fill.Dimname_Scalar(Tensor self, str dim, Tensor index, Scalar value) -> Tensor """
    raise NotImplementedError("torch.ops.aten.index_fill.Dimname_Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill.Dimname_Tensor)
def conveter_aten_index_fill_Dimname_Tensor(
        self: Tensor,
        dim: str,
        index: Tensor,
        value: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index_fill.Dimname_Tensor(Tensor self, str dim, Tensor index, Tensor value) -> Tensor """
    raise NotImplementedError("torch.ops.aten.index_fill.Dimname_Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill.int_Scalar_out)
def conveter_aten_index_fill_int_Scalar_out(
        self: Tensor,
        dim: int,
        index: Tensor,
        value: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index_fill.int_Scalar_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.index_fill.int_Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill.int_Tensor_out)
def conveter_aten_index_fill_int_Tensor_out(
        self: Tensor,
        dim: int,
        index: Tensor,
        value: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index_fill.int_Tensor_out(Tensor self, int dim, Tensor index, Tensor value, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.index_fill.int_Tensor_out ge converter is not implement!")


