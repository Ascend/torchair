import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
from torchair.ge_concrete_graph.utils import dtype_promote
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


@register_fx_node_ge_converter(torch.ops.aten.maximum.default)
def conveter_aten_maximum_default(
        self: Tensor,
        other: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::maximum(Tensor self, Tensor other) -> Tensor """
    self, other = dtype_promote(self, other, target_dtype = meta_outputs.dtype)
    return ge.Maximum(self, other)


@register_fx_node_ge_converter(torch.ops.aten.maximum.out)
def conveter_aten_maximum_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::maximum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.maximum.out ge converter is not implement!")


