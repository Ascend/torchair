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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported([
    Support(F16(4, 8)),
    Support(F32(4, 8)),
])
@register_fx_node_ge_converter(torch.ops.aten.gelu.default)
def conveter_aten_gelu_default(
    self: Tensor, *, approximate: str = "None", meta_outputs: TensorSpec = None
):
    """NB: aten::gelu(Tensor self, *, str approximate="none") -> Tensor"""
    if approximate != "None":
        raise NotImplementedError("gelu only supports approximate is None")
    return ge.Gelu(self)


@register_fx_node_ge_converter(torch.ops.aten.gelu.out)
def conveter_aten_gelu_out(
    self: Tensor,
    *,
    approximate: str = "None",
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::gelu.out(Tensor self, *, str approximate="none", Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.gelu.out ge_converter is not implemented!")
