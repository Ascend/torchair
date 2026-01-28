from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(8, 256))
])
@register_fx_node_ge_converter(torch.ops.npu.npu_dynamic_dual_level_mx_quant.default)
def conveter_npu_dynamic_dual_level_mx_quant_default(
    x: Tensor,
    smooth_scale: Optional[Tensor] = None,
    round_mode: str = "rint",
    meta_outputs: List[TensorSpec] = None
):
    """
    NB: aten::npu_dynamic_dual_level_mx_quant(Tensor input, *, 
                                   Tensor smooth_scale=None, str round_mode="rint") -> (Tensor y, Tensor level0_scale, Tensor level1_scale)
    """
    dst_type = 296
    y, level0_scale, level1_scale = ge.DynamicDualLevelMxQuant(x, smooth_scale, round_mode=round_mode, level0_block_size=512, level1_block_size=32)
    y.desc.dtype = torch_dtype_value_to_ge_proto_type(dst_type)
    level0_scale.desc.dtype = ProtoDataType.DT_FLOAT
    level1_scale.desc.dtype = ProtoDataType.DT_FLOAT8_E8M0

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
    return ge.Reshape(y, y_shape_uint8), level0_scale, level1_scale