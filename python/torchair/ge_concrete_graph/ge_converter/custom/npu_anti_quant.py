from typing import (
    Any,
    Optional
)

import torch
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter, \
                                                       torch_type_to_ge_type
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, DataType
from torchair.ge_concrete_graph.supported_declaration import F32, I8, Support


@declare_supported([
    Support(I8(3, 12, 20, 5), F32(1)),
    Support(I8(3, 12, 17, 5), F32(1), dst_dtype=torch.bfloat16),
    Support(I8(3, 12, 17, 5), F32(1), offset=F32(1), dst_dtype=torch.bfloat16, src_dtype=torch.int8),
    Support(I8(3, 12, 17, 5), F32(1), offset=F32(1)),
    Support(I8(3, 12, 17, 5), F32(1), offset=F32(1), src_dtype=torch.int8),
    Support(I8(3, 12, 17, 5), F32(1), dst_dtype=torch.bfloat16, src_dtype=torch.int8),
    Support(I8(3, 12, 17, 5), F32(1), dst_dtype=torch.float16, src_dtype=torch.int8)    
])
@register_fx_node_ge_converter(torch.ops.npu.npu_anti_quant.default)
def convert_npu_anti_quant_default(
    x: Tensor,
    scale: Tensor,
    *,
    offset: Optional[Tensor] = None,
    dst_dtype: Optional[int] = None,
    src_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_anti_quant(Tensor x, Tensor scale, *, Tensor? offset, ScalarType? dst_dtype,
                               ScalarType? src_dtype) -> (Tensor)"""
    if x.dtype != DataType.DT_INT8:
        raise RuntimeError("AntiQuant only support int8 for input x")

    if src_dtype is not None:
        if src_dtype != torch.int8:
            raise NotImplementedError("AntiQuant only support int8 for src_dtype")

        if src_dtype != x.dtype:
            x.set_torch_dtype(src_dtype)

    dst_type = DataType.DT_FLOAT16
    if dst_dtype is not None:
        dst_type = torch_type_to_ge_type(dst_dtype)
    
    return ge.AscendAntiQuantV2(x, scale, offset, dst_type=dst_type, sqrt_mode=False)
