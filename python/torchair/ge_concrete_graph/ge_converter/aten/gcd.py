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


@register_fx_node_ge_converter(torch.ops.aten.gcd.default)
def conveter_aten_gcd_default(
        self: Tensor,
        other: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::gcd(Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.gcd.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.gcd.out)
def conveter_aten_gcd_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::gcd.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.gcd.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.gcd.int)
def conveter_aten_gcd_int(
        a: int,
        b: int,
        meta_outputs: Any = None):
    """ NB: aten::gcd.int(int a, int b) -> int """
    raise NotImplementedError("torch.ops.aten.gcd.int ge converter is not implement!")


