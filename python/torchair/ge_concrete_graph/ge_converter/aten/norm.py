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


@register_fx_node_ge_converter(torch.ops.aten.norm.Scalar)
def conveter_aten_norm_Scalar(
        self: Tensor,
        p: Union[Number, Tensor] = 2,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor """
    raise NotImplementedError("torch.ops.aten.norm.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.norm.ScalarOpt_dim)
def conveter_aten_norm_ScalarOpt_dim(
        self: Tensor,
        p: Optional[Union[Number, Tensor]],
        dim: List[int],
        keepdim: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor """
    raise NotImplementedError("torch.ops.aten.norm.ScalarOpt_dim ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.norm.names_ScalarOpt_dim)
def conveter_aten_norm_names_ScalarOpt_dim(
        self: Tensor,
        p: Optional[Union[Number, Tensor]],
        dim: List[str],
        keepdim: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::norm.names_ScalarOpt_dim(Tensor self, Scalar? p, str[1] dim, bool keepdim=False) -> Tensor """
    raise NotImplementedError("torch.ops.aten.norm.names_ScalarOpt_dim ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.norm.ScalarOpt_dim_dtype)
def conveter_aten_norm_ScalarOpt_dim_dtype(
        self: Tensor,
        p: Optional[Union[Number, Tensor]],
        dim: List[int],
        keepdim: bool,
        *,
        dtype: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor """
    raise NotImplementedError("torch.ops.aten.norm.ScalarOpt_dim_dtype ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.norm.dtype_out)
def conveter_aten_norm_dtype_out(
        self: Tensor,
        p: Optional[Union[Number, Tensor]],
        dim: List[int],
        keepdim: bool,
        *,
        dtype: int,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.norm.dtype_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.norm.out)
def conveter_aten_norm_out(
        self: Tensor,
        p: Optional[Union[Number, Tensor]],
        dim: List[int],
        keepdim: bool = False,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.norm.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.norm.ScalarOpt_dtype)
def conveter_aten_norm_ScalarOpt_dtype(
        self: Tensor,
        p: Optional[Union[Number, Tensor]],
        *,
        dtype: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor """
    raise NotImplementedError("torch.ops.aten.norm.ScalarOpt_dtype ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.norm.ScalarOpt_dtype_out)
def conveter_aten_norm_ScalarOpt_dtype_out(
        self: Tensor,
        p: Optional[Union[Number, Tensor]],
        *,
        dtype: int,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::norm.ScalarOpt_dtype_out(Tensor self, Scalar? p, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.norm.ScalarOpt_dtype_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.norm.Scalar_out)
def conveter_aten_norm_Scalar_out(
        self: Tensor,
        p: Union[Number, Tensor] = 2,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::norm.Scalar_out(Tensor self, Scalar p=2, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.norm.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.norm.names_ScalarOpt_dim_dtype)
def conveter_aten_norm_names_ScalarOpt_dim_dtype(
        self: Tensor,
        p: Optional[Union[Number, Tensor]],
        dim: List[str],
        keepdim: bool,
        *,
        dtype: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, str[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor """
    raise NotImplementedError("torch.ops.aten.norm.names_ScalarOpt_dim_dtype ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.norm.names_dtype_out)
def conveter_aten_norm_names_dtype_out(
        self: Tensor,
        p: Optional[Union[Number, Tensor]],
        dim: List[str],
        keepdim: bool,
        *,
        dtype: int,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::norm.names_dtype_out(Tensor self, Scalar? p, str[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.norm.names_dtype_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.norm.names_out)
def conveter_aten_norm_names_out(
        self: Tensor,
        p: Optional[Union[Number, Tensor]],
        dim: List[str],
        keepdim: bool = False,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::norm.names_out(Tensor self, Scalar? p, str[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.norm.names_out ge converter is not implement!")


