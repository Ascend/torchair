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
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(5)),
    Support(F16(5)),
    Support(I64(5)),
    Support(I32(5)),
    Support(I16(5)),
    Support(I8(5)),
])
@register_fx_node_ge_converter(torch.ops.aten.log2.default)
def conveter_aten_log2_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::log2(Tensor self) -> Tensor"""
    return ge.Log(self, base=2)


@register_fx_node_ge_converter(torch.ops.aten.log2.out)
def conveter_aten_log2_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.log2.out ge_converter is not implemented!")
