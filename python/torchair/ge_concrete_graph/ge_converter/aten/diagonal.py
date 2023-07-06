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


@register_fx_node_ge_converter(torch.ops.aten.diagonal.default)
def conveter_aten_diagonal_default(
        self: Tensor,
        offset: int = 0,
        dim1: int = 0,
        dim2: int = 1,
        meta_outputs: Any = None):
    """ NB: aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a) """
    raise NotImplementedError("torch.ops.aten.diagonal.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.diagonal.Dimname)
def conveter_aten_diagonal_Dimname(
        self: Tensor,
        *,
        outdim: str,
        dim1: str,
        dim2: str,
        offset: int = 0,
        meta_outputs: Any = None):
    """ NB: aten::diagonal.Dimname(Tensor(a) self, *, str outdim, str dim1, str dim2, int offset=0) -> Tensor(a) """
    raise NotImplementedError("torch.ops.aten.diagonal.Dimname ge converter is not implement!")


