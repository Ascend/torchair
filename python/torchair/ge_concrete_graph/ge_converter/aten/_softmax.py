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
from torchair.ge_concrete_graph.utils import dtype_promote

@declare_supported([
    Support(F32(32, 100, 100), -1, False),
    Support(F32(32, 100, 100), 1, False),
])
@register_fx_node_ge_converter(torch.ops.aten._softmax.default)
def conveter_aten__softmax_default(
    self: Tensor, dim: int, half_to_float: bool, meta_outputs: TensorSpec = None
):
    """NB: aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"""
    if half_to_float:
        raise NotImplementedError("torch.ops.aten._softmax.default unsupport param!")
    return ge.SoftmaxV2(self, axes=[dim])


@register_fx_node_ge_converter(torch.ops.aten._softmax.out)
def conveter_aten__softmax_out(
    self: Tensor,
    dim: int,
    half_to_float: bool,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._softmax.out ge_converter is not implemented!")
