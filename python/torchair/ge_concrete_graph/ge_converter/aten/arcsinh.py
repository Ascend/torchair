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


@register_fx_node_ge_converter(torch.ops.aten.arcsinh.default)
def conveter_aten_arcsinh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::arcsinh(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.arcsinh.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.arcsinh.out)
def conveter_aten_arcsinh_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::arcsinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.arcsinh.out ge_converter is not implemented!")
