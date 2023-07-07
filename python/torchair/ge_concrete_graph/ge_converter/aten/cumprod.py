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


@register_fx_node_ge_converter(torch.ops.aten.cumprod.default)
def conveter_aten_cumprod_default(
        self: Tensor,
        dim: int,
        *,
        dtype: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.cumprod.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cumprod.dimname)
def conveter_aten_cumprod_dimname(
        self: Tensor,
        dim: str,
        *,
        dtype: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cumprod.dimname(Tensor self, str dim, *, ScalarType? dtype=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.cumprod.dimname ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cumprod.dimname_out)
def conveter_aten_cumprod_dimname_out(
        self: Tensor,
        dim: str,
        *,
        dtype: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cumprod.dimname_out(Tensor self, str dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.cumprod.dimname_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cumprod.out)
def conveter_aten_cumprod_out(
        self: Tensor,
        dim: int,
        *,
        dtype: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.cumprod.out ge converter is not implement!")


