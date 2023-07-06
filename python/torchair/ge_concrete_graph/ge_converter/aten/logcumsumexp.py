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


@register_fx_node_ge_converter(torch.ops.aten.logcumsumexp.default)
def conveter_aten_logcumsumexp_default(
        self: Tensor,
        dim: int,
        meta_outputs: Any = None):
    """ NB: aten::logcumsumexp(Tensor self, int dim) -> Tensor """
    raise NotImplementedError("torch.ops.aten.logcumsumexp.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.logcumsumexp.dimname)
def conveter_aten_logcumsumexp_dimname(
        self: Tensor,
        dim: str,
        meta_outputs: Any = None):
    """ NB: aten::logcumsumexp.dimname(Tensor self, str dim) -> Tensor """
    raise NotImplementedError("torch.ops.aten.logcumsumexp.dimname ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.logcumsumexp.dimname_out)
def conveter_aten_logcumsumexp_dimname_out(
        self: Tensor,
        dim: str,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::logcumsumexp.dimname_out(Tensor self, str dim, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.logcumsumexp.dimname_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.logcumsumexp.out)
def conveter_aten_logcumsumexp_out(
        self: Tensor,
        dim: int,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.logcumsumexp.out ge converter is not implement!")


