
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
@register_fx_node_ge_converter(torch.ops.aten.size.int)
def conveter_aten_size_int(
        self: Tensor,
        dim: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::size.int(Tensor self, int dim) -> int """
    raise NotImplementedError("torch.ops.aten.size.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.size.Dimname)
def conveter_aten_size_Dimname(
        self: Tensor,
        dim: str,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::size.Dimname(Tensor self, str dim) -> int """
    raise NotImplementedError("torch.ops.aten.size.Dimname ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.size.default)
def conveter_aten_size_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::size(Tensor self) -> int[] """
    raise NotImplementedError("torch.ops.aten.size.default ge converter is not implement!")


