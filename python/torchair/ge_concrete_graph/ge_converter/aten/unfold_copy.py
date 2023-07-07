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


@register_fx_node_ge_converter(torch.ops.aten.unfold_copy.default)
def conveter_aten_unfold_copy_default(
        self: Tensor,
        dimension: int,
        size: int,
        step: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::unfold_copy(Tensor self, int dimension, int size, int step) -> Tensor """
    raise NotImplementedError("torch.ops.aten.unfold_copy.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.unfold_copy.out)
def conveter_aten_unfold_copy_out(
        self: Tensor,
        dimension: int,
        size: int,
        step: int,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::unfold_copy.out(Tensor self, int dimension, int size, int step, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.unfold_copy.out ge converter is not implement!")


