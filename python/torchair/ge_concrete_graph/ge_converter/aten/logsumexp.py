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


@register_fx_node_ge_converter(torch.ops.aten.logsumexp.default)
def conveter_aten_logsumexp_default(
        self: Tensor,
        dim: List[int],
        keepdim: bool = False,
        meta_outputs: Any = None):
    """ NB: aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor """
    raise NotImplementedError("torch.ops.aten.logsumexp.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.logsumexp.names)
def conveter_aten_logsumexp_names(
        self: Tensor,
        dim: List[str],
        keepdim: bool = False,
        meta_outputs: Any = None):
    """ NB: aten::logsumexp.names(Tensor self, str[1] dim, bool keepdim=False) -> Tensor """
    raise NotImplementedError("torch.ops.aten.logsumexp.names ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.logsumexp.names_out)
def conveter_aten_logsumexp_names_out(
        self: Tensor,
        dim: List[str],
        keepdim: bool = False,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::logsumexp.names_out(Tensor self, str[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.logsumexp.names_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.logsumexp.out)
def conveter_aten_logsumexp_out(
        self: Tensor,
        dim: List[int],
        keepdim: bool = False,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.logsumexp.out ge converter is not implement!")


