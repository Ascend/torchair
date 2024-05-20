import math
import torch
from torch.library import Library, impl

m = Library("npu_inference", "DEF")
m.define("npu_tome_merge(Tensor token_a, Tensor token_b, Tensor topk_indice, \
         Tensor arg_max, float top_rate) -> (Tensor, Tensor, Tensor)")
m.define("npu_tome_unmerge(Tensor atten_out, Tensor ori_indice_a, Tensor ori_indice_b, \
         Tensor topk_indice, Tensor arg_max, float top_rate) -> Tensor")


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