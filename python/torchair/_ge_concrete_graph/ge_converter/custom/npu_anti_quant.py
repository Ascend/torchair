from typing import (
    Any,
    Optional
)

import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter, \
                                                       torch_type_to_ge_type
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import F32, I8, Support


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

    if x.dtype == DataType.DT_INT8:
        if all((src_dtype is not None, src_dtype != torch.int8)):
            raise RuntimeError("torch.ops.npu.npu_anti_quant.default: x data type is int8, "
                f"src_dtype should be same to x, but current src_dtype is {src_dtype}")
    elif x.dtype == DataType.DT_INT32:
        if all((src_dtype is not None, src_dtype != torch.quint4x2)):
            raise RuntimeError("torch.ops.npu.npu_anti_quant.default: x data type is int32, "
                f"src_dtype should be torch.quint4x2, but current src_dtype is {src_dtype}")

        x_dim_num = x.rank
        if x_dim_num == 0:
            raise RuntimeError(f"torch.ops.npu.npu_anti_quant.default: when x data type is int32, "
                "AntiQuant no support for x is scalar")

        bit_shape = [1, ] * (x_dim_num - 1) + [8, ]
        const = ge.Const(bit_shape)
        x_shape_int32 = ge.Shape(x)
        x_shape_int4 = ge.Mul(x_shape_int32, const)
        x = ge.Bitcast(x, type=DataType.DT_INT4)
        x = ge.Reshape(x, x_shape_int4)
    else:
        raise RuntimeError("torch.ops.npu.npu_anti_quant.default: AntiQuant only support int8 or int32 for input x")

    attr_dst_type = DataType.DT_FLOAT16
    if dst_dtype is not None:
        attr_dst_type = torch_type_to_ge_type(dst_dtype)
    
    return ge.AscendAntiQuantV2(x, scale, offset=offset, dst_type=attr_dst_type, sqrt_mode=False)
