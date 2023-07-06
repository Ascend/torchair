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


@register_fx_node_ge_converter(torch.ops.aten.complex.default)
def conveter_aten_complex_default(
        real: Tensor,
        imag: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::complex(Tensor real, Tensor imag) -> Tensor """
    raise NotImplementedError("torch.ops.aten.complex.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.complex.out)
def conveter_aten_complex_out(
        real: Tensor,
        imag: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::complex.out(Tensor real, Tensor imag, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.complex.out ge converter is not implement!")


