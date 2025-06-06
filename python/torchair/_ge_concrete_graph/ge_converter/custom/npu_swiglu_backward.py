from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, \
    I64, I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2, 2), F32(2, 2)),
    Support(F16(2, 2), F16(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_swiglu_backward.default)
def conveter_aten_swiglu_backward_default(
        y_grad: Tensor,
        x: Tensor,
        dim: int = -1,
        meta_outputs: TensorSpec = None):
    """ NB: aten::swiglu_backward(Tensor y_grad, Tensor self) -> Tensor """
    return ge.SwiGluGrad(y_grad, x, dim=dim)
