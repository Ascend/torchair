from torchair.ge._ge_graph import (
    Tensor,
    TensorSpec,
    DataType,
    torch_dtype_value_to_ge_type,
    torch_dtype_value_to_ge_proto_type,
)
from torchair._ge_concrete_graph.ge_converter.converter_utils import (
    ge,
    torch,
    register_fx_node_ge_converter,
    Optional,
    List,
)


def unpack_uint8_to_float4_e2m1(x: Tensor, x_dtype: int) -> Tensor:
    target_dtype = torch_dtype_value_to_ge_type(x_dtype)
    x_const = ge.Const([1] * (x.rank - 1) + [2])
    x_shape = ge.Shape(x)
    x_new_shape = ge.Mul(x_shape, x_const)
    x = ge.Bitcast(x, type=target_dtype)
    x.desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype)
    x = ge.Reshape(x, x_new_shape)
    return x


def pack_int4_to_uint8(tensor: Tensor) -> Tensor:
    bit_shape = [1, 2]
    div_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
    y_shape_int4 = ge.Shape(tensor)
    y_shape_uint8 = ge.Div(y_shape_int4, div_x2)
    y_shape_int4_2bit = ge.ConcatV2(
        [y_shape_uint8, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)], concat_dim=0, N=2
    )
    y = ge.Bitcast(ge.Reshape(tensor, y_shape_int4_2bit), type=DataType.DT_UINT8)
    expanded_x = ge.Reshape(y, y_shape_uint8)
    return expanded_x


@register_fx_node_ge_converter(torch.ops.npu.npu_moe_init_routing_v2.default)
def conveter_npu_moe_init_routing_v2_default(
    x: Tensor,
    expert_idx: Tensor,
    *,
    scale: Optional[Tensor] = None,
    offset: Optional[Tensor] = None,
    active_num: int = -1,
    expert_capacity: int = -1,
    expert_num: int = -1,
    drop_pad_mode: int = 0,
    expert_tokens_num_type: int = 0,
    expert_tokens_num_flag: bool = False,
    quant_mode: int = -1,
    active_expert_range: Optional[List[int]] = None,
    row_idx_type: int = 0,
    x_dtype: Optional[int] = None,
    meta_outputs: List[TensorSpec] = None,
):
    if x_dtype is not None:
        if x.dtype == DataType.DT_UINT8 and torch_dtype_value_to_ge_type(x_dtype) == DataType.DT_FLOAT4_E2M1:
            x = unpack_uint8_to_float4_e2m1(x, x_dtype)
        else:
            x = ge.Bitcast(x, type=torch_dtype_value_to_ge_type(x_dtype))
            x.desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype)
    if scale is not None and quant_mode == -1:
        if x.dtype in (DataType.DT_FLOAT8_E4M3FN, DataType.DT_FLOAT8_E5M2, DataType.DT_FLOAT4_E2M1):
            scale = ge.Bitcast(scale, type=DataType.DT_FLOAT8_E8M0)
    expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale = ge.MoeInitRoutingV3(
        x,
        expert_idx,
        scale,
        offset,
        active_num=active_num,
        expert_capacity=expert_capacity,
        expert_num=expert_num,
        drop_pad_mode=drop_pad_mode,
        expert_tokens_num_type=expert_tokens_num_type,
        expert_tokens_num_flag=expert_tokens_num_flag,
        quant_mode=quant_mode,
        active_expert_range=active_expert_range,
        row_idx_type=row_idx_type,
    )
    if x_dtype is not None:
        if torch_dtype_value_to_ge_type(x_dtype) == DataType.DT_FLOAT4_E2M1:
            expanded_x = pack_int4_to_uint8(expanded_x)
        else:
            expanded_x.desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype)
    elif quant_mode in [6, 7, 8]:
        import torch_npu

        expanded_x.desc.dtype = torch_dtype_value_to_ge_proto_type(torch_npu.hifloat8)
    elif quant_mode == 9:
        expanded_x = pack_int4_to_uint8(expanded_x)
    return expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale
