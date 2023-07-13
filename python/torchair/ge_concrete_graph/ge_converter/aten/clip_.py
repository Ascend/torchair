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


@register_fx_node_ge_converter(torch.ops.aten.clip_.default)
def conveter_aten_clip__default(
        self: Tensor,
        min: Optional[Union[Number, Tensor]] = None,
        max: Optional[Union[Number, Tensor]] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::clip_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.clip_.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.clip_.Tensor)
def conveter_aten_clip__Tensor(
        self: Tensor,
        min: Optional[Tensor] = None,
        max: Optional[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::clip_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.clip_.Tensor ge converter is not implement!")


