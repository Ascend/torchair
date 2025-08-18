from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_dequant_rope_quant_kvcache.default)
def conveter_aten_dequant_rope_quant_kvcache_default(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    k_cache_ref: Tensor,
    v_cache_ref: Tensor,
    indices: Tensor,
    scale_k: Tensor,
    scale_v: Tensor,
    size_splits: List[int],
    offset_k: Optional[Tensor] = None,
    offset_v: Optional[Tensor] = None,
    weight_scale: Optional[Tensor] = None,
    activation_scale: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    quant_mode: int = 0,
    input_layout: str = "BSND",
    kv_output: bool = False,
    cache_mode="contiguous",
    meta_outputs: List[TensorSpec] = None,
):
    quant_mode_str = "static" if quant_mode == 0 else "dynamic"
    return ge.DequantRopeQuantKvcache(
        x,
        cos,
        sin,
        k_cache_ref,
        v_cache_ref,
        indices,
        scale_k,
        scale_v,
        offset_k,
        offset_v,
        weight_scale,
        activation_scale,
        bias,
        size_splits=size_splits,
        quant_mode=quant_mode_str,
        layout=input_layout,
        kv_output=kv_output,
        cache_mode=cache_mode,
    )
