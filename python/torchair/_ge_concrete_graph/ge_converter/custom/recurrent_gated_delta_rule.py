from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
])
@register_fx_node_ge_converter(torch.ops.npu.npu_recurrent_gated_delta_rule.default)
def conveter_npu_recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    state: Tensor,
    *,
    beta: Optional[Tensor] = None,
    scale: Optional[float] = None,
    actual_seq_lengths: Optional[Tensor] = None,
    ssm_state_indices: Optional[Tensor] = None,
    num_accepted_tokens: Optional[Tensor] = None,
    g: Optional[Tensor] = None,
    gk: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_recurrent_gated_delta_rule(Tensor query, Tensor key, Tensor value, Tensor(a!) state, *, 
                                               Tensor? beta=None, float? scale=None, Tensor? actual_seq_lengths=None, 
                                               Tensor? ssm_state_indices=None, Tensor? num_accepted_tokens=None, 
                                               Tensor? g=None, Tensor? gk=None) -> Tensor
    """
    return ge.RecurrentGatedDeltaRule(
        query,
        key,
        value,
        beta,
        state,
        actual_seq_lengths,
        ssm_state_indices,
        g=g,
        gk=gk,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale)


@declare_supported([
])
@register_fx_node_ge_converter(torch.ops.npu.npu_recurrent_gated_delta_rule_functional.default)
def conveter_npu_recurrent_gated_delta_rule_functional(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    state: Tensor,
    *,
    beta: Optional[Tensor] = None,
    scale: Optional[float] = None,
    actual_seq_lengths: Optional[Tensor] = None,
    ssm_state_indices: Optional[Tensor] = None,
    num_accepted_tokens: Optional[Tensor] = None,
    g: Optional[Tensor] = None,
    gk: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None
):
    """
    NB: npu::npu_recurrent_gated_delta_rule_functional(Tensor query, Tensor key, Tensor value, Tensor state, *, 
                                                       Tensor? beta=None, float? scale=None, Tensor? actual_seq_lengths=None, 
                                                       Tensor? ssm_state_indices=None, Tensor? num_accepted_tokens=None, 
                                                       Tensor? g=None, Tensor? gk=None) -> (Tensor, Tensor)
    """
    state_copy = ge.TensorMove(state)
    return ge.RecurrentGatedDeltaRule(
        query,
        key,
        value,
        beta,
        state_copy,
        actual_seq_lengths,
        ssm_state_indices,
        g=g,
        gk=gk,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale)
