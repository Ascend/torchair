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
from torch import Generator, contiguous_format, inf, memory_format, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.elu.default)
def conveter_aten_elu_default(
    self: Tensor,
    alpha: Union[Number, Tensor] = 1,
    scale: Union[Number, Tensor] = 1,
    input_scale: Union[Number, Tensor] = 1,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.elu.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.elu.out)
def conveter_aten_elu_out(
    self: Tensor,
    alpha: Union[Number, Tensor] = 1,
    scale: Union[Number, Tensor] = 1,
    input_scale: Union[Number, Tensor] = 1,
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.elu.out ge_converter is not implemented!")
