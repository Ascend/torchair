from torchair._ge_concrete_graph.ge_converter.converter_utils import *


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
    """NB: npu::npu_anti_quant(Tensor x, Tensor scale, *, Tensor? offset, int? dst_dtype,
                               int? src_dtype) -> (Tensor)"""
    INPUT_AND_SCALE_SUPPORT_MAP = {
        DataType.DT_UINT8: DataType.DT_HIFLOAT8,
        DataType.DT_FLOAT8_E4M3FN: DataType.DT_FLOAT8_E4M3FN,
        DataType.DT_FLOAT8_E5M2: DataType.DT_FLOAT8_E5M2,
        DataType.DT_INT8: DataType.DT_INT8,
        DataType.DT_INT32: DataType.DT_INT4
    }

    if x.dtype in INPUT_AND_SCALE_SUPPORT_MAP:
        if src_dtype is not None:
            if x.dtype == DataType.DT_INT32 and src_dtype != torch.quint4x2 and torch_dtype_value_to_ge_type(src_dtype) != DataType.DT_INT4:
                raise RuntimeError("torch.ops.npu.npu_anti_quant.default: "
                    f"when x is Int32, src_dtype must be int4, but current src_dtype is {src_dtype} and x_dtype is {x.dtype}")
            elif x.dtype != DataType.DT_INT32 and torch_dtype_value_to_ge_type(src_dtype) != INPUT_AND_SCALE_SUPPORT_MAP.get(x.dtype):
                raise RuntimeError("torch.ops.npu.npu_anti_quant.default: "
                    f"src_dtype should be match to x, but current src_dtype is {src_dtype} and x_dtype is {x.dtype}")

        if x.dtype == DataType.DT_UINT8:
            x = ge.Bitcast(x, type=DataType.DT_HIFLOAT8)
        elif x.dtype == DataType.DT_INT32:
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
        raise RuntimeError("torch.ops.npu.npu_anti_quant.default: AntiQuant only support int8, int32, hifloat8, float8_e5m2 or float8_e4m3fn for input x")

    acl_dst_type = DataType.DT_FLOAT16
    if dst_dtype is not None:
        acl_dst_type = torch_dtype_value_to_ge_type(dst_dtype)
    
    return ge.AscendAntiQuantV2(x, scale, offset=offset, dst_type=acl_dst_type, sqrt_mode=False)
