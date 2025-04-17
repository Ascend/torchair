from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.gcd.default)
def conveter_aten_gcd_default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::gcd(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.gcd.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gcd.out)
def conveter_aten_gcd_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::gcd.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.gcd.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gcd.int)
def conveter_aten_gcd_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::gcd.int(int a, int b) -> int"""
    raise NotImplementedError("torch.ops.aten.gcd.int ge_converter is not implemented!")
