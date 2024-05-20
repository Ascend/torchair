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
from torchair._ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.arccosh.default)
def conveter_aten_arccosh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::arccosh(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.arccosh.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.arccosh.out)
def conveter_aten_arccosh_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::arccosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.arccosh.out ge_converter is not implemented!")