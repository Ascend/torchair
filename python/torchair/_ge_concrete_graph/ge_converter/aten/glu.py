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
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F32(64, 4, 9), 1),
        Support(F32(6, 5, 8), 2),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.glu.default)
def conveter_aten_glu_default(self: Tensor, dim: int = -1, meta_outputs: TensorSpec = None):
    """NB: aten::glu(Tensor self, int dim=-1) -> Tensor"""
    if self.rank < 1:
        raise NotImplementedError("torch.ops.aten.glu.default does not support 0-dimensional tensor")
    return ge.GLU(self, dim=dim)


@register_fx_node_ge_converter(torch.ops.aten.glu.out)
def conveter_aten_glu_out(
    self: Tensor, dim: int = -1, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::glu.out(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.glu.out ge_converter is not supported!")
