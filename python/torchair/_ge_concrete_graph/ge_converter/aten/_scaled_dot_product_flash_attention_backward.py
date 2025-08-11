from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._scaled_dot_product_flash_attention_backward.default)
def conveter_aten__scaled_dot_product_flash_attention_backward_default(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    cum_seq_q: Tensor,
    cum_seq_k: Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: Tensor,
    philox_offset: Tensor,
    *,
    scale: Optional[float] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_scaled_dot_product_flash_attention_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, int max_q, int max_k, float dropout_p, bool is_causal, Tensor philox_seed, Tensor philox_offset, *, float? scale=None) -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)"""
    raise NotImplementedError(
        "torch.ops.aten._scaled_dot_product_flash_attention_backward.default ge_converter is not implemented!"
    )
