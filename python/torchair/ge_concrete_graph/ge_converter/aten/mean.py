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


@register_fx_node_ge_converter(torch.ops.aten.mean.default)
def conveter_aten_mean_default(
        self: Tensor,
        *,
        dtype: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.mean.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mean.dim)
def conveter_aten_mean_dim(
        self: Tensor,
        dim: Optional[List[int]],
        keepdim: bool = False,
        *,
        dtype: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor """
    if dtype is not None:
        # TODO: fix this case
        print(f"[warning] torch.ops.aten.mean.dim have some unprocessed parameters or cases!")

    return ge.ReduceMean(self, dim, keep_dims=keepdim)


@register_fx_node_ge_converter(torch.ops.aten.mean.names_dim)
def conveter_aten_mean_names_dim(
        self: Tensor,
        dim: List[str],
        keepdim: bool = False,
        *,
        dtype: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::mean.names_dim(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.mean.names_dim ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mean.names_out)
def conveter_aten_mean_names_out(
        self: Tensor,
        dim: List[str],
        keepdim: bool = False,
        *,
        dtype: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::mean.names_out(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.mean.names_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mean.out)
def conveter_aten_mean_out(
        self: Tensor,
        dim: Optional[List[int]],
        keepdim: bool = False,
        *,
        dtype: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::mean.out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.mean.out ge converter is not implement!")


