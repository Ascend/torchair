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


@register_fx_node_ge_converter(torch.ops.aten.clamp_.default)
def conveter_aten_clamp__default(
        self: Tensor,
        min: Optional[Union[Number, Tensor]] = None,
        max: Optional[Union[Number, Tensor]] = None,
        meta_outputs: Any = None):
    """ NB: aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.clamp_.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_.Tensor)
def conveter_aten_clamp__Tensor(
        self: Tensor,
        min: Optional[Tensor] = None,
        max: Optional[Tensor] = None,
        meta_outputs: Any = None):
    """ NB: aten::clamp_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.clamp_.Tensor ge converter is not implement!")


