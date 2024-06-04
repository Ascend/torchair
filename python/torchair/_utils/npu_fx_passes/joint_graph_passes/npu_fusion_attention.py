import sys
import functools
import logging
import math

import torch
import torch.nn as nn

try:
    import torch_npu
except ImportError:
    pass

from torch._dynamo.utils import counters
from torch._inductor.pattern_matcher import (
    filter_nodes,
    register_replacement,
    _return_true
)

try:
    from torch._inductor.pattern_matcher import inference_graph, training_graph
except ImportError:
    from torch._inductor.pattern_matcher import fwd_only as inference_graph
    from torch._inductor.pattern_matcher import joint_fwd_bwd as training_graph
    
from torch._inductor.fx_passes.joint_graph import patterns
from torchair._utils.npu_fx_passes.joint_graph import register_joint_graph_pass
from torchair.core.utils import logger

log = logging.getLogger(__name__)
aten = torch.ops.aten


def rotary_emb(cos_cached, sin_cached, x, seq_len=None):
    return (
        cos_cached[:seq_len].to(dtype=x.dtype),
        sin_cached[:seq_len].to(dtype=x.dtype),
    )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _npu_fusion_attention_pattern_1(query_states, key_states, value_states, attention_mask, head_dim):
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=0, training=True)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output


def _npu_fusion_attention_replace_1(query_states, key_states, value_states, attention_mask, head_dim):
    counters["inductor"]["npu_fusion_attention"] += 1
    _, num_heads, _, _ = query_states.size()
    norm_factor = math.sqrt(head_dim)
    attention_mask = attention_mask.to(torch.bool)
    hidden_size = num_heads * head_dim
    attn_output = torch_npu.npu_fusion_attention(
        query_states,
        key_states,
        value_states,
        num_heads,
        "BNSD",
        pse=None,
        padding_mask=None,
        atten_mask=attention_mask,
        scale=1.0 / norm_factor,
        pre_tockens=hidden_size,
        next_tockens=0,
        keep_prob=1,
    )[0]
    return attn_output



def _npu_fusion_attention_pattern_2(
    query_states,
    key_states,
    value_states,
    attention_mask,
    position_ids,
    cos_cached,
    sin_cached,
    num_heads,
    head_dim
):
    bsz, q_len, hidden_size = query_states.size()
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    num_key_value_heads = num_heads
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    kv_seq_len = q_len
    cos, sin = rotary_emb(cos_cached, sin_cached, value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=0, training=True)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    return attn_output


def _npu_fusion_attention_replace_2(
    query_states,
    key_states,
    value_states,
    attention_mask,
    position_ids,
    cos_cached,
    sin_cached,
    num_heads,
    head_dim
):
    counters["inductor"]["npu_fusion_attention"] += 1
    bsz, q_len, hidden_size = query_states.size()
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    num_key_value_heads = num_heads
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    kv_seq_len = q_len
    cos, sin = rotary_emb(cos_cached, sin_cached, value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    norm_factor = math.sqrt(head_dim)
    attention_mask = attention_mask.to(torch.bool)
    attn_output = torch_npu.npu_fusion_attention(
        query_states,
        key_states,
        value_states,
        num_heads,
        "BNSD",
        pse=None,
        padding_mask=None,
        atten_mask=attention_mask,
        scale=1.0 / norm_factor,
        pre_tockens=hidden_size,
        next_tockens=0,
        keep_prob=1,
    )[0]

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    return attn_output


def _get_npu_fusion_attention_candidates():
    device = "npu"
    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds
    q1 = functools.partial(torch.empty, (2, 32, 2048, 128), dtype=torch.bfloat16, device=device, requires_grad=True)
    k1 = functools.partial(torch.empty, (2, 32, 2048, 128), dtype=torch.bfloat16, device=device, requires_grad=True)
    v1 = functools.partial(torch.empty, (2, 32, 2048, 128), dtype=torch.bfloat16, device=device, requires_grad=True)
    am1 = functools.partial(torch.empty, (2, 1, 2048, 2048), dtype=torch.float32, device=device, requires_grad=False)
    d1 = {"head_dim":128}
    candidates = [
            ( 
                _npu_fusion_attention_pattern_1,
                _npu_fusion_attention_replace_1,
                [q1(), k1(), v1(), am1()], 
                d1,
                _return_true,
            )
        ]
    
    qkv = functools.partial(torch.empty, (2, 2048, 4096), dtype=torch.bfloat16, device=device, requires_grad=True)
    position_ids = functools.partial(torch.empty, (1, 2048), dtype=torch.int64, device=device, requires_grad=False)
    am2 = functools.partial(torch.empty, (2, 1, 2048, 2048), device=device, requires_grad=False)
    cos_cached = functools.partial(torch.empty, (4096, 128), device=device, requires_grad=False)
    sin_cached = functools.partial(torch.empty, (4096, 128), device=device, requires_grad=False)
    d2 = {"num_heads":32, "head_dim":128}
    for dtype in [torch.float32, torch.bfloat16]:
        am2_dtype = functools.partial(am2, dtype=dtype)
        cos_cached_dtype = functools.partial(cos_cached, dtype=dtype)
        sin_cached_dtype = functools.partial(sin_cached, dtype=dtype)
        candidates.append(
                (
                    _npu_fusion_attention_pattern_2,
                    _npu_fusion_attention_replace_2,
                    [qkv(), qkv(), qkv(), am2_dtype(), position_ids(), cos_cached_dtype(), sin_cached_dtype()], 
                    d2,
                    _return_true,
                ),
            )
    return candidates

        
@register_joint_graph_pass('npu_fusion_attention')
@functools.lru_cache(None)
def _npu_fusion_attention_init():
    if 'torch_npu' not in sys.modules:
        logger.info(f'The npu_fusion_attention fx pass will only be enabled in a torch npu env.'
                    'When there is no torch_npu in the env, skip npu_fusion_attention fx pass.')
        return
    candidates = _get_npu_fusion_attention_candidates()
    
    for pattern, replacement, args, workaround, extra_check in candidates:
        args = [*args, *workaround.values()]
        register_replacement(
            pattern,
            replacement,
            args,
            training_graph,
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
        )