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


@register_fx_node_ge_converter(torch.ops.aten.dot.default)
def conveter_aten_dot_default(self: Tensor, tensor: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::dot(Tensor self, Tensor tensor) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.dot.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.dot.out)
def conveter_aten_dot_out(
    self: Tensor, tensor: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.dot.out ge_converter is not implemented!")
