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


def unpack_uint8_to_float4(tokens: Tensor, tokens_dtype: int) -> Tensor:
    target_dtype = torch_dtype_value_to_ge_type(tokens_dtype)
    tokens_const = ge.Const([1] * (tokens.rank - 1) + [2])
    tokens_shape = ge.Shape(tokens)
    tokens_new_shape = ge.Mul(tokens_shape, tokens_const)
    tokens = ge.Bitcast(tokens, type=target_dtype)
    tokens.desc.dtype = torch_dtype_value_to_ge_proto_type(tokens_dtype)
    tokens = ge.Reshape(tokens, tokens_new_shape)
    return tokens


def pack_int4_to_uint8(tensor: Tensor) -> Tensor:
    bit_shape = [1, 2]
    div_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
    y_shape_int4 = ge.Shape(tensor)
    y_shape_uint8 = ge.Div(y_shape_int4, div_x2)
    y_shape_int4_2bit = ge.ConcatV2(
        [y_shape_uint8, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)], concat_dim=0, N=2
    )
    y = ge.Bitcast(ge.Reshape(tensor, y_shape_int4_2bit), type=DataType.DT_UINT8)
    permute_tokens = ge.Reshape(y, y_shape_uint8)
    return permute_tokens


def is_fp4_dtype(dtype: int) -> bool:
    return torch_dtype_value_to_ge_type(dtype) in (DataType.DT_FLOAT4_E2M1, DataType.DT_FLOAT4_E1M2)


@register_fx_node_ge_converter(torch.ops.npu.npu_moe_re_routing.default)
def conveter_npu_moe_re_routing_default(
        tokens: Tensor,
        expert_token_num_per_rank: Tensor,
        *,
        per_token_scales: Optional[Tensor] = None,
        expert_token_num_type: int = 1,
        idx_type: int = 0,
        tokens_dtype: Optional[int] = None,
        meta_outputs: List[TensorSpec] = None):
    if tokens_dtype is not None:
        if tokens.dtype == DataType.DT_UINT8 and is_fp4_dtype(tokens_dtype):
            tokens = unpack_uint8_to_float4(tokens, tokens_dtype)
        else:
            tokens = ge.Bitcast(tokens, type=torch_dtype_value_to_ge_type(tokens_dtype))
            tokens.desc.dtype = torch_dtype_value_to_ge_proto_type(tokens_dtype)

    permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num = \
        ge.MoeReRouting(tokens, expert_token_num_per_rank, per_token_scales,
                        expert_token_num_type=expert_token_num_type,
                        idx_type=idx_type)

    if tokens_dtype is not None:
        if is_fp4_dtype(tokens_dtype):
            permute_tokens = pack_int4_to_uint8(permute_tokens)
        else:
            permute_tokens.desc.dtype = torch_dtype_value_to_ge_proto_type(tokens_dtype)

    return permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num
