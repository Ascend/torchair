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
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F32(1), offset=None)
    ]
)

@register_fx_node_ge_converter(torch.ops.npu.npu_trans_quant_param.default)
def conveter_npu_npu_trans_quant_param(
    scale: Tensor,
    offset: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_trans_quant_param(Tensor scale, Tensor? offset=None) -> Tensor
    """
    return ge.TransQuantParamV2(scale, offset=offset)
