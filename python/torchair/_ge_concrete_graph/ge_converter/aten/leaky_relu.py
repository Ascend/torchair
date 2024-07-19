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
    Support(F32(2, 2), 0.01),
])
@register_fx_node_ge_converter(torch.ops.aten.leaky_relu.default)
def conveter_aten_leaky_relu_default(
    self: Tensor, negative_slope: Union[Number, Tensor] = 0.01, meta_outputs: TensorSpec = None
):
    """NB: aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor"""
    if isinstance(negative_slope, Tensor):
        raise RuntimeError(
            "Follow the same implementation as the community, torch.ops.aten.leaky_relu.default "
            "only supports negative_slope is Number now.")
    return ge.LeakyRelu(self, negative_slope=negative_slope)

@register_fx_node_ge_converter(torch.ops.aten.leaky_relu.out)
def conveter_aten_leaky_relu_out(
    self: Tensor,
    negative_slope: Union[Number, Tensor] = 0.01,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError(
        "torch.ops.aten.leaky_relu.out is redundant before pytorch 2.1.0,might be supported in future version.")
