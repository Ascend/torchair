import math
from typing import Optional
import torch
from torch.library import Library, impl

m = Library("npu_inference", "DEF")
m.define("npu_tome_merge(Tensor token_a, Tensor token_b, Tensor topk_indice, \
         Tensor arg_max, float top_rate) -> (Tensor, Tensor, Tensor)")
m.define("npu_tome_unmerge(Tensor atten_out, Tensor ori_indice_a, Tensor ori_indice_b, \
         Tensor topk_indice, Tensor arg_max, float top_rate) -> Tensor")
m.define("npu_moe_gating_top_k(Tensor x, int k, *, Tensor? bias=None, int k_group=1, \
         int group_count=1, int group_select_mode=0, int renorm=0, int norm_type=0, \
         bool y2_flag=False, float routed_scaling_factor=1.0, float eps=1e-20) \
         -> (Tensor, Tensor, Tensor)")
m.define("npu_kv_rmsnorm_rope_cache(Tensor kv, Tensor gamma, Tensor cos, Tensor sin, \
          Tensor index, Tensor k_cache, Tensor v_cache, float epsilon=1e-5) -> (Tensor, Tensor)")
m.define("npu_interleave_rope(Tensor x, Tensor cos, Tensor sin) -> Tensor")
m.define("npu_dequant_swiglu_quant(Tensor x, Tensor? weight_scale, Tensor? activate_scale, \
          Tensor? bias, Tensor? quant_scale, Tensor? quant_offset, Tensor? group_index, \
          bool activate_left=False, int quant_mode=0) -> (Tensor, Tensor)")


@impl(m, "npu_tome_merge", "PrivateUse1")
def plug_npu_tome_merge(
        token_a: torch.Tensor,
        token_b: torch.Tensor,
        topk_indice: torch.Tensor,
        arg_max: torch.Tensor,
        top_rate: float
):
    return token_a, token_b, arg_max


@impl(m, "npu_tome_merge", "Meta")
def npu_tome_merge_meta(token_a, token_b, topk_indice, arg_max, top_rate=0.5):
    batch = token_a.size(0)
    seq_len_a = token_a.size(1)
    hidden_size = token_a.size(2)
    seq_len_b = token_b.size(1)
    top_r = math.floor((seq_len_a + seq_len_b) * top_rate)
    heads = 8
    unmerge_token_a_dim_list = [batch, seq_len_a - top_r, hidden_size]
    unmerge_token_b_dim_list = [batch, heads, seq_len_b, hidden_size]
    unreduce_count_dim_list = [batch, heads, seq_len_b]
    unreduce_count = torch.empty(unreduce_count_dim_list, dtype=torch.float32, device='meta')
    return (token_a.new_empty(tuple(unmerge_token_a_dim_list)), token_a.new_empty(tuple(unmerge_token_b_dim_list)),
            torch.empty_like(unreduce_count))


@impl(m, "npu_tome_unmerge", "PrivateUse1")
def plug_npu_tome_unmerge(
        atten_out: torch.Tensor,
        ori_indice_a: torch.Tensor,
        ori_indice_b: torch.Tensor,
        topk_indice: torch.Tensor,
        arg_max: torch.Tensor,
        top_r_rate: float
):
    return atten_out


@impl(m, "npu_tome_unmerge", "Meta")
def npu_tome_unmerge_meta(atten_out, ori_indice_a, ori_indice_b, topk_indice, arg_max, top_r_rate=0.5):
    dim_list = []
    dim_list.append(atten_out.size(0))
    dim_list.append(ori_indice_a.size(1) + ori_indice_b.size(1))
    dim_list.append(atten_out.size(2))
    return atten_out.new_empty(tuple(dim_list))


@impl(m, "npu_moe_gating_top_k", "PrivateUse1")
def plug_npu_moe_gating_top_k(
        x: torch.Tensor,
        k: int,
        *,
        bias: Optional[torch.Tensor] = None,
        k_group: int = 1,
        group_count: int = 1,
        group_select_mode: int = 0,
        renorm: int = 0,
        norm_type: int = 0,
        y2_flag: bool = False,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20,
):
    return x


@impl(m, "npu_moe_gating_top_k", "Meta")
def npu_moe_gating_top_k(x, k, *, bias=None, k_group=1, group_count=1, group_select_mode=0, renorm=0,
                         norm_type=0, y2_flag=False, routed_scaling_factor=1.0, eps=1e-20):
    x_dim = x.dim()
    if bias is not None:
        bias_dim = bias.dim()
    y_dim_list = [x.size(0), k]
    expert_idx_dim_list = [x.size(0), k]
    y2_dim_list = [x.size(0), x.size(1)]
    return (x.new_empty(tuple(y_dim_list), dtype=x.dtype),
            x.new_empty(tuple(expert_idx_dim_list), dtype=torch.int32),
            x.new_empty(tuple(y_dim_list), dtype=torch.float32))


@impl(m, "npu_kv_rmsnorm_rope_cache", "PrivateUse1")
def plug_npu_kv_rmsnorm_rope_cache(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    index: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    epsilon: float
):
    return k_cache, v_cache


@impl(m, "npu_kv_rmsnorm_rope_cache", "Meta")
def npu_kv_rmsnorm_rope_cache_meta(kv, gamma, cos, sin, index, k_cache, v_cache, epsilon=1e-5):
    return torch.empty_like(k_cache), torch.empty_like(v_cache)


@impl(m, "npu_interleave_rope", "PrivateUse1")
def plug_npu_interleave_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
):
    return x


@impl(m, "npu_interleave_rope", "Meta")
def npu_interleave_rope_meta(x, cos, sin):
    return torch.empty_like(x)


@impl(m, "npu_dequant_swiglu_quant", "PrivateUse1")
def plug_npu_dequant_swiglu_quant(
    x: torch.Tensor,
    weight_scale: torch.Tensor = None,
    activation_scale: torch.Tensor = None,
    bias: torch.Tensor = None,
    quant_scale: torch.Tensor = None,
    quant_offset: torch.Tensor = None,
    group_index: torch.Tensor = None,
    activate_left: bool = False,
    quant_mode: int = 0
):
    return x


@impl(m, "npu_dequant_swiglu_quant", "Meta")
def npu_dequant_swiglu_quant_meta(x, weight_scale=None, activation_scale=None, bias=None, quant_scale=None,
                                  quant_offset=None, group_index=None, activate_left=False, quant_mode=0):
    y_size = []
    scale_size = []
    for i in range(x.dim() - 1):
        y_size.append(x.size(i))
        scale_size.append(x.size(i))
    y_size.append(math.floor(x.size(x.dim() - 1) / 2))
    return (torch.empty(y_size, dtype=torch.int8, device=x.device),
            torch.empty(scale_size, dtype=torch.float32, device=x.device))
