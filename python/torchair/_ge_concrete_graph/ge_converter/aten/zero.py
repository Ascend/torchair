from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import F32, Support


@declare_supported(
    [
        Support(F32(8, 8)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.zero.default)
def conveter_aten_zero_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::zero(Tensor self) -> Tensor"""
    return ge.ZerosLike(self)


@register_fx_node_ge_converter(torch.ops.aten.zero.out)
def conveter_aten_zero_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::zero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.zero.out ge_converter is not implemented!")
