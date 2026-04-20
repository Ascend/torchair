from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(8, 256))
])
@register_fx_node_ge_converter(torch.ops.npu.npu_dynamic_block_mx_quant.default)
def conveter_npu_dynamic_block_mx_quant_default(
    x: Tensor,
    round_mode: str = "rint",
    dst_type: int = 296,
    scale_alg: int = 0,
    dst_type_max: float = 0.0,
    meta_outputs: List[TensorSpec] = None
):
    """
    NB: aten::npu_dynamic_block_mx_quant(Tensor x, 
                                         *, 
                                         str round_mode="rint",
                                         int dst_type=torch_npu.float4_e2m1
                                         int scale_alg=0,
                                         float dst_type_max=0.0) -> (Tensor y, Tensor scale1, Tensor scale2)
    """
    acl_dst_type = torch_dtype_value_to_ge_type(dst_type)
    y, scale1, scale2 = ge.DynamicBlockMxQuant(x, round_mode=round_mode, dst_type=acl_dst_type, scale_alg=scale_alg, dst_type_max=dst_type_max)
    y.desc.dtype = torch_dtype_value_to_ge_proto_type(dst_type)
    scale1.desc.dtype = ProtoDataType.DT_FLOAT8_E8M0
    scale2.desc.dtype = ProtoDataType.DT_FLOAT8_E8M0
    if dst_type == 296 or dst_type == 297:
        dim_num = x.rank
        bit_shape = []
        for _ in range(dim_num - 1):
            bit_shape.append(1)
        bit_shape.append(2)
        div_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
        y_shape_int4 = ge.Shape(y)
        y_shape_uint8 = ge.Div(y_shape_int4, div_x2)
        y_shape_int4_2bit = ge.ConcatV2([y_shape_uint8, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)],
                                        concat_dim=0, N=2)
        y = ge.Bitcast(ge.Reshape(y, y_shape_int4_2bit), type=DataType.DT_UINT8)
        return ge.Reshape(y, y_shape_uint8), scale1, scale2
    else:
        return y, scale1, scale2
