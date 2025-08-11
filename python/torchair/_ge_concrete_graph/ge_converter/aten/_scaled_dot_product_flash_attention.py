from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._scaled_dot_product_flash_attention.default)
def conveter_aten__scaled_dot_product_flash_attention_default(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_scaled_dot_product_flash_attention(Tensor query, Tensor key, Tensor value, float dropout_p=0., bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor ouput, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, int max_q, int max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)"""
    raise NotImplementedError(
        "torch.ops.aten._scaled_dot_product_flash_attention.default ge_converter is not implemented!"
    )
