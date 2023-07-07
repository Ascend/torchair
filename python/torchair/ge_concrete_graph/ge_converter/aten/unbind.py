import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
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


@register_fx_node_ge_converter(torch.ops.aten.unbind.int)
def conveter_aten_unbind_int(
        self: Tensor,
        dim: int = 0,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[] """
    raise NotImplementedError("torch.ops.aten.unbind.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.unbind.Dimname)
def conveter_aten_unbind_Dimname(
        self: Tensor,
        dim: str,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::unbind.Dimname(Tensor(a -> *) self, str dim) -> Tensor(a)[] """
    raise NotImplementedError("torch.ops.aten.unbind.Dimname ge converter is not implement!")


