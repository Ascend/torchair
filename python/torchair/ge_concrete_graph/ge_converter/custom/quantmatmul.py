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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(I8(8192, 320), I8(320, 2560), F32(1), offset=None, bias=None, transpose_x1=False, transpose_x2=False)
    ]
)

@register_fx_node_ge_converter(torch.ops.npu.npu_quant_matmul.default)
def conveter_npu_npu_quant_matmul(
    x1: Tensor,
    x2: Tensor,
    scale: Tensor,
    offset: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_quant_matmul(Tensor x1, Tensor x2, Tensor scale, Tensor? offset=None, Tensor? bias=None) -> Tensor
    """
    return ge.QuantBatchMatmulV3(x1, x2, scale, offset=offset, bias=bias, transpose_x1=False, transpose_x2=False)
