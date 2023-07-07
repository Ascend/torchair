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


@register_fx_node_ge_converter(torch.ops.aten.multiply_.Tensor)
def conveter_aten_multiply__Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::multiply_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.multiply_.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.multiply_.Scalar)
def conveter_aten_multiply__Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::multiply_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.multiply_.Scalar ge converter is not implement!")


