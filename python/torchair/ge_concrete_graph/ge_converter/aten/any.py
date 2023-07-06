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


@register_fx_node_ge_converter(torch.ops.aten.any.default)
def conveter_aten_any_default(
        self: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::any(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.any.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.any.dim)
def conveter_aten_any_dim(
        self: Tensor,
        dim: int,
        keepdim: bool = False,
        meta_outputs: Any = None):
    """ NB: aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor """
    raise NotImplementedError("torch.ops.aten.any.dim ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.any.out)
def conveter_aten_any_out(
        self: Tensor,
        dim: int,
        keepdim: bool = False,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.any.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.any.all_out)
def conveter_aten_any_all_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::any.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.any.all_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.any.dimname)
def conveter_aten_any_dimname(
        self: Tensor,
        dim: str,
        keepdim: bool = False,
        meta_outputs: Any = None):
    """ NB: aten::any.dimname(Tensor self, str dim, bool keepdim=False) -> Tensor """
    raise NotImplementedError("torch.ops.aten.any.dimname ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.any.dimname_out)
def conveter_aten_any_dimname_out(
        self: Tensor,
        dim: str,
        keepdim: bool = False,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::any.dimname_out(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.any.dimname_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.any.str)
def conveter_aten_any_str(
        self: List[str],
        meta_outputs: Any = None):
    """ NB: aten::any.str(str[] self) -> bool """
    raise NotImplementedError("torch.ops.aten.any.str ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.any.int)
def conveter_aten_any_int(
        self: List[int],
        meta_outputs: Any = None):
    """ NB: aten::any.int(int[] self) -> bool """
    raise NotImplementedError("torch.ops.aten.any.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.any.float)
def conveter_aten_any_float(
        self: List[float],
        meta_outputs: Any = None):
    """ NB: aten::any.float(float[] self) -> bool """
    raise NotImplementedError("torch.ops.aten.any.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.any.bool)
def conveter_aten_any_bool(
        self: List[bool],
        meta_outputs: Any = None):
    """ NB: aten::any.bool(bool[] self) -> bool """
    raise NotImplementedError("torch.ops.aten.any.bool ge converter is not implement!")


