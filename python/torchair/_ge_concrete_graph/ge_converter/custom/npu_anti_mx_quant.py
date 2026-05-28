from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_anti_mx_quant.default)
def conveter_npu_anti_mx_quant_default(
    x: Tensor,
    mxscale: Tensor,
    *,
    axis: int = -1,
    dst_type: int = 15, # torch.bfloat16 enum value is 15
    src_type: int = 292, # torch.float8_e4m3fn enum value is 292
    meta_outputs: TensorSpec = None
):
    """
    NB: aten::npu_anti_mx_quant(Tensor x,
                                Tensor mxscale, *,
                                int axis=-1,
                                int dst_type=torch_npu.bfloat16) -> Tensor
    """
    ge_dst_type = torch_dtype_value_to_ge_type(dst_type)
    ge_src_type = torch_dtype_value_to_ge_type(src_type)
    if ge_dst_type not in [DataType.DT_FLOAT16, DataType.DT_FLOAT, DataType.DT_BF16]:
        raise RuntimeError("Parameter dst_type only supports torch.float16, torch.float32, torch.bfloat16, "
                           "got " + str(dst_type))
    if x.rank < 1 or x.rank > 7:
        raise RuntimeError(f"Input x dimNum should be between 1 and 7, got {x.rank}")
    if mxscale.rank < 2 or mxscale.rank > 8:
        raise RuntimeError(f"Input mxscale dimNum should be between 2 and 8, got {mxscale.rank}")
    if axis < -x.rank or axis >= x.rank:
        raise RuntimeError("Parameter axis is out of x dimension range, got " + str(axis))

    if x.dtype == DataType.DT_UINT8 and src_type != 296 and src_type != 297:
        raise RuntimeError("torch.ops.npu.npu_anti_mx_quant.default: when x is uint8, src_type must be float4")
    elif (x.dtype != torch_dtype_value_to_ge_type(src_type)):
        raise RuntimeError("src_type should be match to x dtype")

    if x.dtype == DataType.DT_UINT8:
        dim_num = x.rank
        bit_shape = [1, ] * (dim_num - 1) + [2, ]
        const = ge.Const(bit_shape)
        x_shape_float8 = ge.Shape(x)
        x_shape_float4 = ge.Mul(x_shape_float8, const)
        x = ge.Bitcast(x, type=ge_src_type)
        x = ge.Reshape(x, x_shape_float4)

    y = ge.AntiMxQuant(x, mxscale, axis=axis, dst_type=ge_dst_type)
    y.desc.dtype = torch_dtype_value_to_ge_proto_type(dst_type)

    return y
