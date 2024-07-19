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
    Support(F32(2, 6, 7, 7), F32(2, 6, 7, 7))
])
@register_fx_node_ge_converter(torch.ops.aten.hardswish_backward.default)
def conveter_aten_hardswish_backward_default(
    grad_output: Tensor, self: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::hardswish_backward(Tensor grad_output, Tensor self) -> Tensor"""
    return ge.HardSwishGrad(grad_output, self)


@register_fx_node_ge_converter(torch.ops.aten.hardswish_backward.out)
def conveter_aten_hardswish_backward_out(
    grad_output: Tensor, self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::hardswish_backward.out(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.hardswish_backward.out ge_converter is redundant before pytorch 2.1.0,"
                       "might be supported in future version.")
