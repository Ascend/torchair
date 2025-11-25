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
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._utils.check_platform import is_arch35


@register_fx_node_ge_converter(torch.ops.npu.npu_fused_matmul.default)
def conveter_npu_npu_fused_matmul(
    x: Tensor,
    x2: Tensor,
    bias: Optional[Tensor] = None,
    x3: Optional[Tensor] = None,
    fused_op_type: Optional[str] = "",
    meta_outputs: TensorSpec = None,
):
    """NB: npu::npu_fused_matmul(Tensor x, Tensor x2, *, Tensor? bias=None, Tensor? x3=None, str fused_op_type='') -> Tensor
    """
    if not is_arch35():
        return
    return ge.FusedMatMul(x, x2, bias=bias, x3=x3, transpose_x1=False, transpose_x2=False, 
                          enable_hf32=torch.npu.matmul.allow_hf32, fused_op_type=fused_op_type)