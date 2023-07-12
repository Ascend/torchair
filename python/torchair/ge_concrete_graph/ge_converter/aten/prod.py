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


@register_fx_node_ge_converter(torch.ops.aten.prod.default)
def conveter_aten_prod_default(
        self: Tensor,
        *,
        dtype: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.prod.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.prod.dim_int)
def conveter_aten_prod_dim_int(
        self: Tensor,
        dim: int,
        keepdim: bool = False,
        *,
        dtype: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.prod.dim_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.prod.dim_Dimname)
def conveter_aten_prod_dim_Dimname(
        self: Tensor,
        dim: str,
        keepdim: bool = False,
        *,
        dtype: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::prod.dim_Dimname(Tensor self, str dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.prod.dim_Dimname ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.prod.Dimname_out)
def conveter_aten_prod_Dimname_out(
        self: Tensor,
        dim: str,
        keepdim: bool = False,
        *,
        dtype: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::prod.Dimname_out(Tensor self, str dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.prod.Dimname_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.prod.int_out)
def conveter_aten_prod_int_out(
        self: Tensor,
        dim: int,
        keepdim: bool = False,
        *,
        dtype: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.prod.int_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.prod.out)
def conveter_aten_prod_out(
        self: Tensor,
        *,
        dtype: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::prod.out(Tensor self, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.prod.out ge converter is not implement!")


