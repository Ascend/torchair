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


@register_fx_node_ge_converter(torch.ops.aten.sum.dim_IntList)
def conveter_aten_sum_dim_IntList(
        self: Tensor,
        dim: Optional[List[int]],
        keepdim: bool = False,
        *,
        dtype: Optional[int] = None,
        meta_outputs: Any = None):
    """ NB: aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.sum.dim_IntList ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sum.default)
def conveter_aten_sum_default(
        self: Tensor,
        *,
        dtype: Optional[int] = None,
        meta_outputs: Any = None):
    """ NB: aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.sum.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sum.dim_DimnameList)
def conveter_aten_sum_dim_DimnameList(
        self: Tensor,
        dim: List[str],
        keepdim: bool = False,
        *,
        dtype: Optional[int] = None,
        meta_outputs: Any = None):
    """ NB: aten::sum.dim_DimnameList(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.sum.dim_DimnameList ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sum.DimnameList_out)
def conveter_aten_sum_DimnameList_out(
        self: Tensor,
        dim: List[str],
        keepdim: bool = False,
        *,
        dtype: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::sum.DimnameList_out(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.sum.DimnameList_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sum.IntList_out)
def conveter_aten_sum_IntList_out(
        self: Tensor,
        dim: Optional[List[int]],
        keepdim: bool = False,
        *,
        dtype: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.sum.IntList_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sum.out)
def conveter_aten_sum_out(
        self: Tensor,
        *,
        dtype: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::sum.out(Tensor self, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.sum.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sum.int)
def conveter_aten_sum_int(
        self: List[int],
        meta_outputs: Any = None):
    """ NB: aten::sum.int(int[] self) -> int """
    raise NotImplementedError("torch.ops.aten.sum.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sum.float)
def conveter_aten_sum_float(
        self: List[float],
        meta_outputs: Any = None):
    """ NB: aten::sum.float(float[] self) -> float """
    raise NotImplementedError("torch.ops.aten.sum.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sum.complex)
def conveter_aten_sum_complex(
        self: List[complex],
        meta_outputs: Any = None):
    """ NB: aten::sum.complex(complex[] self) -> complex """
    raise NotImplementedError("torch.ops.aten.sum.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sum.bool)
def conveter_aten_sum_bool(
        self: List[bool],
        meta_outputs: Any = None):
    """ NB: aten::sum.bool(bool[] self) -> int """
    raise NotImplementedError("torch.ops.aten.sum.bool ge converter is not implement!")


