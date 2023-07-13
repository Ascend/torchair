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


@register_fx_node_ge_converter(torch.ops.aten.lerp.Scalar)
def conveter_aten_lerp_Scalar(
        self: Tensor,
        end: Tensor,
        weight: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor """
    raise NotImplementedError("torch.ops.aten.lerp.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lerp.Tensor)
def conveter_aten_lerp_Tensor(
        self: Tensor,
        end: Tensor,
        weight: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor """
    raise NotImplementedError("torch.ops.aten.lerp.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lerp.Scalar_out)
def conveter_aten_lerp_Scalar_out(
        self: Tensor,
        end: Tensor,
        weight: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.lerp.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lerp.Tensor_out)
def conveter_aten_lerp_Tensor_out(
        self: Tensor,
        end: Tensor,
        weight: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.lerp.Tensor_out ge converter is not implement!")


