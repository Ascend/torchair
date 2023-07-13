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


@register_fx_node_ge_converter(torch.ops.aten.clamp.default)
def conveter_aten_clamp_default(
        self: Tensor,
        min: Optional[Union[Number, Tensor]] = None,
        max: Optional[Union[Number, Tensor]] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.clamp.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.clamp.Tensor)
def conveter_aten_clamp_Tensor(
        self: Tensor,
        min: Optional[Tensor] = None,
        max: Optional[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.clamp.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.clamp.out)
def conveter_aten_clamp_out(
        self: Tensor,
        min: Optional[Union[Number, Tensor]] = None,
        max: Optional[Union[Number, Tensor]] = None,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.clamp.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.clamp.Tensor_out)
def conveter_aten_clamp_Tensor_out(
        self: Tensor,
        min: Optional[Tensor] = None,
        max: Optional[Tensor] = None,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::clamp.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.clamp.Tensor_out ge converter is not implement!")


