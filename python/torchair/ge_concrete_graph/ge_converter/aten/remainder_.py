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


@register_fx_node_ge_converter(torch.ops.aten.remainder_.Tensor)
def conveter_aten_remainder__Tensor(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.remainder_.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.remainder_.Scalar)
def conveter_aten_remainder__Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.remainder_.Scalar ge_converter is not implemented!")
