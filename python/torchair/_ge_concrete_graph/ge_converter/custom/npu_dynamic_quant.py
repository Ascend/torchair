from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(64, 16384, 16384)),
    Support(BF16(64, 16384, 16384)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_dynamic_quant.default)
def conveter_npu_dynamic_quant_default(
    input_data: Tensor,
    smooth_scales: Optional[Tensor] = None,
    group_index: Optional[Tensor] = None,
    dst_type: int = 1,
    quant_mode: str = "pertoken",
    meta_outputs: List[TensorSpec] = None
):
    acl_dst_type = torch_dtype_value_to_ge_type(dst_type)

    if quant_mode == "pertoken":
        y, scale = ge.DynamicQuant(input_data, smooth_scales=smooth_scales, group_index=group_index, dst_type=acl_dst_type)
    else:
        y, scale, offset = ge.DynamicQuantV2(input_data, smooth_scales,
                                             group_index, dst_type=acl_dst_type,
                                             is_symmetrical=True, quant_mode=quant_mode)
    if dst_type == 16: # torch.quint4x2
        dim_num = input_data.rank
        bit_shape = []
        for _ in range(dim_num - 1):
            bit_shape.append(1)
        bit_shape.append(8)
        div_x2 = ge.Const(bit_shape, dtype=DataType.DT_INT32)
        y_shape_int4 = ge.Shape(y)
        y_shape_int32 = ge.Div(y_shape_int4, div_x2)
        y_shape_int4_4bit = ge.ConcatV2([y_shape_int32, ge.Const([8], dtype=DataType.DT_INT32)],
                                        concat_dim=0, N=2)
        y = ge.Bitcast(ge.Reshape(y, y_shape_int4_4bit), type=DataType.DT_INT32)
        y = ge.Reshape(y, y_shape_int32)
    return y, scale


@declare_supported([
    Support(F16(64, 16384, 16384)),
    Support(BF16(64, 16384, 16384)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_dynamic_quant_asymmetric.default)
def conveter_npu_dynamic_quant_asymmetric_default(
    input_data: Tensor,
    smooth_scales: Optional[Tensor] = None,
    group_index: Optional[Tensor] = None,
    dst_type: int = 1,
    quant_mode: str = "pertoken",
    meta_outputs: List[TensorSpec] = None
):
    acl_dst_type = torch_dtype_value_to_ge_type(dst_type)
    y, scale, offset = ge.DynamicQuantV2(input_data, smooth_scales,
                                         group_index, dst_type=acl_dst_type,
                                         is_symmetrical=False, quant_mode=quant_mode)
    if dst_type == 16:
        dim_num = input_data.rank
        bit_shape = []
        for _ in range(dim_num - 1):
            bit_shape.append(1)
        bit_shape.append(8)
        div_x2 = ge.Const(bit_shape, dtype=DataType.DT_INT32)
        y_shape_int4 = ge.Shape(y)
        y_shape_int32 = ge.Div(y_shape_int4, div_x2)
        y_shape_int4_4bit = ge.ConcatV2([y_shape_int32, ge.Const([8], dtype=DataType.DT_INT32)],
                                        concat_dim=0, N=2)
        y = ge.Bitcast(ge.Reshape(y, y_shape_int4_4bit), type=DataType.DT_INT32)
        y = ge.Reshape(y, y_shape_int32)
    return y, scale, offset
