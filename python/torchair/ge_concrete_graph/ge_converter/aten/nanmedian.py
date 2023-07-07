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


@register_fx_node_ge_converter(torch.ops.aten.nanmedian.default)
def conveter_aten_nanmedian_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::nanmedian(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.nanmedian.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.nanmedian.dim)
def conveter_aten_nanmedian_dim(
        self: Tensor,
        dim: int,
        keepdim: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::nanmedian.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices) """
    raise NotImplementedError("torch.ops.aten.nanmedian.dim ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.nanmedian.dim_values)
def conveter_aten_nanmedian_dim_values(
        self: Tensor,
        dim: int,
        keepdim: bool = False,
        *,
        values: Tensor = None,
        indices: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::nanmedian.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices) """
    raise NotImplementedError("torch.ops.aten.nanmedian.dim_values ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.nanmedian.names_dim)
def conveter_aten_nanmedian_names_dim(
        self: Tensor,
        dim: str,
        keepdim: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::nanmedian.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices) """
    raise NotImplementedError("torch.ops.aten.nanmedian.names_dim ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.nanmedian.names_dim_values)
def conveter_aten_nanmedian_names_dim_values(
        self: Tensor,
        dim: str,
        keepdim: bool = False,
        *,
        values: Tensor = None,
        indices: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::nanmedian.names_dim_values(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices) """
    raise NotImplementedError("torch.ops.aten.nanmedian.names_dim_values ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.nanmedian.out)
def conveter_aten_nanmedian_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::nanmedian.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.nanmedian.out ge converter is not implement!")


