from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.leaky_relu.default)
def conveter_aten_leaky_relu_default(
    self: Tensor, negative_slope: Union[Number, Tensor] = 0.01, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.leaky_relu.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.leaky_relu.out)
def conveter_aten_leaky_relu_out(
    self: Tensor,
    negative_slope: Union[Number, Tensor] = 0.01,
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.leaky_relu.out ge_converter is not implemented!")
