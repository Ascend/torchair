from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._scaled_dot_product_efficient_attention_backward.default)
def conveter_aten__scaled_dot_product_efficient_attention_backward_default(
    grad_out_: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    is_causal: bool = False,
    chunk_grad_outputs: bool = False,
    *,
    scale: Optional[float] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_scaled_dot_product_efficient_attention_backward(Tensor grad_out_, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, bool is_causal=False, bool chunk_grad_outputs=False, *, float? scale=None) -> (Tensor, Tensor, Tensor)"""
    raise NotImplementedError(
        "torch.ops.aten._scaled_dot_product_efficient_attention_backward.default ge_converter is not implemented!"
    )
