from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_fused_causal_conv1d_functional.default)
def converter_npu_fused_causal_conv1d_functional(
        x: Tensor,
        weight: Tensor,
        conv_states: Tensor,
        *,
        query_start_loc: Optional[Tensor] = None,
        cache_indices: Optional[Tensor] = None,
        initial_state_mode: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        num_accepted_tokens: Optional[Tensor] = None,
        activation_mode: Optional[str] = "None",
        pad_slot_id: Optional[int] = -1,
        run_mode: Optional[int] = 0,
        residual_connection: Optional[int] = 0,
        meta_outputs: TensorSpec = None):
    """
    NB: func: npu_fused_causal_conv1d_functional(Tensor x, Tensor weight, Tensor conv_states, *, Tensor? query_start_loc=None, 
                                                Tensor? cache_indices=None, Tensor? initial_state_mode=None, Tensor? bias=None, 
                                                Tensor? num_accepted_tokens=None, str? activation_mode="None", int? pad_slot_id=-1, 
                                                int? run_mode=0, int? residual_connection=0) -> (Tensor, Tensor)

    """
    activation_value = 0
    if activation_mode == "silu":
        activation_value = 1
    elif activation_mode == "swish":
        activation_value = 2
    conv_states_copy = ge.TensorMove(conv_states)
    return ge.FusedCausalConv1d(
        x=x,
        weight=weight,
        conv_states=conv_states_copy,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        initial_state_mode=initial_state_mode,
        bias=bias,
        num_accepted_tokens=num_accepted_tokens,
        activation_mode=activation_value,
        pad_slot_id=pad_slot_id,
        run_mode=run_mode,
        residual_connection=residual_connection
    )
