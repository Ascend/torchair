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


@register_fx_node_ge_converter(torch.ops.aten.clamp_min.default)
def conveter_aten_clamp_min_default(
        self: Tensor,
        min: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::clamp_min(Tensor self, Scalar min) -> Tensor """
    raise NotImplementedError("torch.ops.aten.clamp_min.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_min.Tensor)
def conveter_aten_clamp_min_Tensor(
        self: Tensor,
        min: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::clamp_min.Tensor(Tensor self, Tensor min) -> Tensor """
    raise NotImplementedError("torch.ops.aten.clamp_min.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_min.out)
def conveter_aten_clamp_min_out(
        self: Tensor,
        min: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.clamp_min.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_min.Tensor_out)
def conveter_aten_clamp_min_Tensor_out(
        self: Tensor,
        min: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::clamp_min.Tensor_out(Tensor self, Tensor min, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.clamp_min.Tensor_out ge converter is not implement!")


