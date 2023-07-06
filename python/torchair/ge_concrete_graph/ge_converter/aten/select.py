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


@register_fx_node_ge_converter(torch.ops.aten.select.Dimname)
def conveter_aten_select_Dimname(
        self: Tensor,
        dim: str,
        index: int,
        meta_outputs: Any = None):
    """ NB: aten::select.Dimname(Tensor(a) self, str dim, int index) -> Tensor(a) """
    raise NotImplementedError("torch.ops.aten.select.Dimname ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.select.int)
def conveter_aten_select_int(
        self: Tensor,
        dim: int,
        index: Union[int, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a) """
    raise NotImplementedError("torch.ops.aten.select.int ge converter is not implement!")


