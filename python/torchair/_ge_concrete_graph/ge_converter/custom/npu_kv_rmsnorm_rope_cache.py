from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_kv_rmsnorm_rope_cache.default)
def conveter_npu_kv_rmsnorm_rope_cache_default(
    kv: Tensor,
    gamma: Tensor,
    cos: Tensor,
    sin: Tensor,
    index: Tensor,
    k_cache: Tensor,
    ckv_cache: Tensor,
    *,
    k_rope_scale: Optional[Tensor] = None,
    c_kv_scale: Optional[Tensor] = None,
    k_rope_offset: Optional[Tensor] = None,
    c_kv_offset: Optional[Tensor] = None,
    epsilon: float = 1e-5,
    cache_mode: str = 'Norm',
    is_output_kv: bool = False,
    meta_outputs: List[TensorSpec] = None

):
    return ge.KvRmsNormRopeCache(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                 k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                                 k_rope_offset=k_rope_offset, c_kv_offset=c_kv_offset,
                                 epsilon=epsilon, cache_mode=cache_mode, is_output_kv=is_output_kv)


@register_fx_node_ge_converter(torch.ops.npu.npu_kv_rmsnorm_rope_cache_v2_functional.default)
def conveter_npu_kv_rmsnorm_rope_cache_v2_functional_default(
    kv: Tensor,
    gamma: Tensor,
    cos: Tensor,
    sin: Tensor,
    index: Tensor,
    k_cache: Tensor,
    ckv_cache: Tensor,
    *,
    k_rope_scale: Optional[Tensor] = None,
    c_kv_scale: Optional[Tensor] = None,
    k_rope_offset: Optional[Tensor] = None,
    c_kv_offset: Optional[Tensor] = None,
    epsilon: float = 1e-5,
    cache_mode: str = 'Norm',
    is_output_kv: bool = False,
    meta_outputs: List[TensorSpec] = None

):
    """
    func: npu_kv_rmsnorm_rope_cache_v2_functional(Tensor kv, Tensor gamma, Tensor cos, Tensor sin, Tensor index, 
                                                  Tensor k_cache, Tensor ckv_cache, *, Tensor? k_rope_scale=None, 
                                                  Tensor? c_kv_scale=None, Tensor? k_rope_offset=None, 
                                                  Tensor? c_kv_offset=None, float epsilon=1e-5, str cache_mode='Norm', 
                                                  bool is_output_kv=False) -> (Tensor, Tensor, Tensor, Tensor)
    """
    # 保证非原地算子的输入不会被修改
    k_cache_copy = ge.TensorMove(k_cache)
    ckv_cache_copy = ge.TensorMove(ckv_cache)
    out0, out1, out2, out3 = ge.KvRmsNormRopeCache(kv, gamma, cos, sin, index, k_cache_copy, ckv_cache_copy,
                                                   k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                                                   k_rope_offset=k_rope_offset, c_kv_offset=c_kv_offset,
                                                   epsilon=epsilon, cache_mode=cache_mode, is_output_kv=is_output_kv)
    return out2, out3, out0, out1