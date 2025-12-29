from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(8, 256))
])
@register_fx_node_ge_converter(torch.ops.npu.npu_dynamic_mx_quant_with_dual_axis.default)
def conveter_npu_dynamic_mx_quant_with_dual_axis_default(
    x: Tensor,
    round_mode: str = "rint",
    dst_type: int = 296,
    scale_alg: int = 0,
    meta_outputs: List[TensorSpec] = None
):
    """
    NB: aten::npu_dynamic_mx_quant_with_dual_axis(Tensor x, *, 
                                                    str round_mode="rint",
                                                    int dst_type=torch_npu.float4_e2m1,
                                                    int scale_alg=0) -> (Tensor y1, Tensor mxscale1, Tensor y2, Tensor mxscale2))
    """
    acl_dst_type = torch_dtype_value_to_ge_type(dst_type)
    y1, mxscale1, y2, mxscale2 = ge.DynamicMxQuantWithDualAxis(x, round_mode=round_mode, dst_type=acl_dst_type, scale_alg=scale_alg)
    y1.desc.dtype = torch_dtype_value_to_ge_proto_type(dst_type)
    mxscale1.desc.dtype = ProtoDataType.DT_FLOAT8_E8M0
    y2.desc.dtype = torch_dtype_value_to_ge_proto_type(dst_type)
    mxscale2.desc.dtype = ProtoDataType.DT_FLOAT8_E8M0
    if dst_type == 296 or dst_type == 297:
        dim_num = x.rank
        bit_shape = []
        for _ in range(dim_num - 1):
            bit_shape.append(1)
        bit_shape.append(2)
        div_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
        y1_shape_int4 = ge.Shape(y1)
        y2_shape_int4 = ge.Shape(y2)
        y1_shape_uint8 = ge.Div(y1_shape_int4, div_x2)
        y2_shape_uint8 = ge.Div(y2_shape_int4, div_x2)
        y1_shape_int4_2bit = ge.ConcatV2([y1_shape_uint8, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)],
                                        concat_dim=0, N=2)
        y2_shape_int4_2bit = ge.ConcatV2([y2_shape_uint8, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)],
                                        concat_dim=0, N=2)
        y1 = ge.Bitcast(ge.Reshape(y1, y1_shape_int4_2bit), type=DataType.DT_UINT8)
        y2 = ge.Bitcast(ge.Reshape(y2, y2_shape_int4_2bit), type=DataType.DT_UINT8)
        return ge.Reshape(y1, y1_shape_uint8), mxscale1, ge.Reshape(y2, y2_shape_uint8), mxscale2
    else:
        return y1, mxscale1, y2, mxscale2