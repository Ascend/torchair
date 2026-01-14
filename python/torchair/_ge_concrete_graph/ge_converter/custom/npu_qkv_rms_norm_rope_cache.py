from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_qkv_rms_norm_rope_cache.default)
def conveter_npu_qkv_rms_norm_rope_cache_default(
    qkv: Tensor,
    q_gamma: Tensor,
    k_gamma: Tensor,
    cos: Tensor,
    sin: Tensor,
    index: Tensor,
    q_out: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    qkv_size: List[int],
    head_nums: List[int],
    *,
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
    k_offset: Optional[Tensor] = None,
    v_offset: Optional[Tensor] = None,
    epsilon: float = 1e-6,
    cache_mode: str = 'PA_NZ',
    is_output_qkv: bool = False,
    meta_outputs: List[TensorSpec] = None
):
    return ge.QkvRmsNormRopeCache(qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache, k_scale=k_scale, 
                                  v_scale=v_scale, k_offset=k_offset, v_offset=v_offset, qkv_size=qkv_size, 
                                  head_nums=head_nums, epsilon=epsilon, cache_mode=cache_mode, is_output_qkv=is_output_qkv)


@register_fx_node_ge_converter(torch.ops.npu.npu_qkv_rms_norm_rope_cache_functional.default)
def conveter_npu_qkv_rms_norm_rope_cache_functional_default(
    qkv: Tensor,
    q_gamma: Tensor,
    k_gamma: Tensor,
    cos: Tensor,
    sin: Tensor,
    index: Tensor,
    q_out: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    qkv_size: List[int],
    head_nums: List[int],
    *,
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
    k_offset: Optional[Tensor] = None,
    v_offset: Optional[Tensor] = None,
    epsilon: float = 1e-6,
    cache_mode: str = 'PA_NZ',
    is_output_qkv: bool = False,
    meta_outputs: List[TensorSpec] = None
):
    """
    func: npu_qkv_rms_norm_rope_cache_functional(Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos, Tensor sin, Tensor index, 
                                                Tensor q_out, Tensor k_cache, Tensor v_cache, *, Tensor? k_scale=None, Tensor? v_scale=None, 
                                                Tensor? k_offset=None, Tensor? v_offset=None, int[4] qkv_size, int[3] head_nums, 
                                                float epsilon=1e-6, str cache_mode='PA_NZ', bool is_output_qkv=False) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
    """
    # 保证非原地算子的输入不会被修改
    q_out_copy = ge.TensorMove(q_out)
    k_cache_copy = ge.TensorMove(k_cache)
    v_cache_copy = ge.TensorMove(v_cache)
    out0, out1, out2, out3, out4, out5 = ge.QkvRmsNormRopeCache(qkv, q_gamma, k_gamma, cos, sin, index, q_out_copy, k_cache_copy, 
                                                    v_cache_copy, k_scale=k_scale, v_scale=v_scale, k_offset=k_offset, 
                                                    v_offset=v_offset, qkv_size=qkv_size, head_nums=head_nums,
                                                    epsilon=epsilon, cache_mode=cache_mode, is_output_qkv=is_output_qkv)
    return out3, out4, out5, out0, out1, out2