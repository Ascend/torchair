from torchair._ge_concrete_graph.ge_converter.converter_utils import *


def _parse_activation(activation):
    """将字符串激活模式转换为整数值（PTA str → GE Int）"""
    if activation == "silu":
        return 1
    elif activation == "swish":
        return 2
    return 0


def _parse_conv_mode(conv_mode):
    """将字符串卷积模式转换为整数值（PTA str → GE Int）
    "default" → 0：正常卷积计算
    "pangu"   → 1：盘古模型下卷积计算前k-1个token置零
    """
    if conv_mode == "pangu":
        return 1
    return 0  # "default"

@register_fx_node_ge_converter(torch.ops.npu.npu_fused_causal_conv1d_v2.default)
def converter_npu_fused_causal_conv1d_v2(
        x: Tensor,
        weight: Tensor,
        conv_states: Tensor,
        *,
        query_start_loc: Optional[Tensor] = None,
        cache_indices: Optional[Tensor] = None,
        initial_state_mode: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        num_accepted_tokens: Optional[Tensor] = None,
        activation: Optional[str] = "None",
        pad_slot_id: Optional[int] = -1,
        run_mode: Optional[int] = 0,
        residual_connection: Optional[int] = 0,
        max_query_len: Optional[int] = -1,
        num_computed_tokens: Optional[Tensor] = None,
        block_idx_first_scheduled_token: Optional[Tensor] = None,
        block_idx_last_scheduled_token: Optional[Tensor] = None,
        initial_state_idx: Optional[Tensor] = None,
        block_size: Optional[int] = 128,
        conv_mode: Optional[str] = "default",
        meta_outputs: TensorSpec = None):
    """
    NB: func: npu_fused_causal_conv1d_v2(Tensor(a!) x, Tensor weight, Tensor(b!) conv_states, *,
              Tensor? query_start_loc=None, Tensor? cache_indices=None, Tensor? initial_state_mode=None,
              Tensor? bias=None, Tensor? num_accepted_tokens=None, str? activation="None",
              int? pad_slot_id=-1, int? run_mode=0, int? residual_connection=0, int? max_query_len=-1,
              Tensor? num_computed_tokens=None, Tensor? block_idx_first_scheduled_token=None,
              Tensor? block_idx_last_scheduled_token=None, Tensor? initial_state_idx=None,
              int? block_size=128, str? conv_mode="default") -> ()
    """
    _conv_states_out, x_out = ge.InplaceFusedCausalConv1d(
        x, weight, conv_states,
        query_start_loc, cache_indices, initial_state_mode, bias,
        num_accepted_tokens, num_computed_tokens,
        block_idx_first_scheduled_token, block_idx_last_scheduled_token,
        initial_state_idx,
        activation_mode=_parse_activation(activation),
        pad_slot_id=pad_slot_id,
        run_mode=run_mode,
        max_query_len=max_query_len,
        residual_connection=residual_connection,
        block_size=block_size,
        conv_mode=_parse_conv_mode(conv_mode)
    )
    return x_out
