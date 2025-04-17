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


@register_fx_node_ge_converter(torch.ops.aten.softshrink.default)
def conveter_aten_softshrink_default(
    self: Tensor, lambd: Union[Number, Tensor] = 0.5, meta_outputs: TensorSpec = None
):
    """NB: aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.softshrink.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.softshrink.out)
def conveter_aten_softshrink_out(
    self: Tensor,
    lambd: Union[Number, Tensor] = 0.5,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.softshrink.out ge_converter is not implemented!")
