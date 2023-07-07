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


@register_fx_node_ge_converter(torch.ops.aten.greater_equal_.Scalar)
def conveter_aten_greater_equal__Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::greater_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.greater_equal_.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.greater_equal_.Tensor)
def conveter_aten_greater_equal__Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::greater_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.greater_equal_.Tensor ge converter is not implement!")


