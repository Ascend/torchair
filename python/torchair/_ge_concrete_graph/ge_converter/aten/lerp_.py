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
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.lerp_.Scalar)
def conveter_aten_lerp__Scalar(
    self: Tensor, end: Tensor, weight: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.lerp_.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.lerp_.Tensor)
def conveter_aten_lerp__Tensor(
    self: Tensor, end: Tensor, weight: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.lerp_.Tensor ge_converter is not implemented!")
