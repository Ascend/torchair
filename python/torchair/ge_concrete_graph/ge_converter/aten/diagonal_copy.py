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


@register_fx_node_ge_converter(torch.ops.aten.diagonal_copy.default)
def conveter_aten_diagonal_copy_default(
        self: Tensor,
        offset: int = 0,
        dim1: int = 0,
        dim2: int = 1,
        meta_outputs: Any = None):
    """ NB: aten::diagonal_copy(Tensor self, int offset=0, int dim1=0, int dim2=1) -> Tensor """
    raise NotImplementedError("torch.ops.aten.diagonal_copy.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.diagonal_copy.out)
def conveter_aten_diagonal_copy_out(
        self: Tensor,
        offset: int = 0,
        dim1: int = 0,
        dim2: int = 1,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::diagonal_copy.out(Tensor self, int offset=0, int dim1=0, int dim2=1, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.diagonal_copy.out ge converter is not implement!")


