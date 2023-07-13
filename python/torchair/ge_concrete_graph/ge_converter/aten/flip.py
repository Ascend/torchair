import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided
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


@register_fx_node_ge_converter(torch.ops.aten.flip.default)
def conveter_aten_flip_default(
        self: Tensor,
        dims: List[int],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::flip(Tensor self, int[] dims) -> Tensor """
    raise NotImplementedError("torch.ops.aten.flip.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.flip.out)
def conveter_aten_flip_out(
        self: Tensor,
        dims: List[int],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::flip.out(Tensor self, int[] dims, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.flip.out ge converter is not implement!")


