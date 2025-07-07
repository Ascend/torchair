# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch_npu
from torch_npu.contrib.module import LinearA8W8Quant

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.llama.configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    # bsz, tgt_len = input_ids_shape
    # # FA 替换mask方式
    # mask = torch.full((tgt_len, tgt_len), 1, device=device, dtype=torch.bool)
    # mask_cond = torch.arange(mask.size(-1), device=device)
    # mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    # return mask
    bsz, tgt_len = input_ids_shape
    # FA 替换mask方式
    mask = torch.full((tgt_len, tgt_len), 1, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(
        bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), 1).to(dtype)
    # expanded_mask = mask[:, None, None, :].expand(bsz, 1, 1, src_len)
    # inverted_mask = ~ expanded_mask
    # # FA 替换mask方式
    # return inverted_mask


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self,
                hidden_states,
                residual: Optional[torch.Tensor] = None):
        # input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to(torch.float32)
        # variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # return self.weight * hidden_states.to(input_dtype)
        import torch_npu
        if residual is None:
            return (torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0], hidden_states)
        else:
            y, _, x = torch_npu.npu_add_rms_norm(
                residual, hidden_states, self.weight, self.variance_epsilon)
            return (y, x)


class LlamaAddRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual):
        # input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to(torch.float32)
        # variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # return self.weight * hidden_states.to(input_dtype)
        return torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / \
            (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[
                             None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[
                             None, None, :, :].to(dtype), persistent=False)

    def forward(self, x=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # if seq_len > self.max_seq_len_cached:
        #     self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        if x is None and seq_len is None:
            return (self.cos_cached, self.sin_cached)
        return (
            # self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            # self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.cos_cached.to(dtype=x.dtype),
            self.sin_cached.to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[
                             None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[
                             None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len /
                 self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / \
                (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(self.max_seq_len_cached,
                         device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[
                             None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[
                             None, None, :, :].to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    # cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    # sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    # cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    # sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    if position_ids is not None:
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        # with torch.autograd.profiler.record_function("position_ids1111111"):
        f_position_ids = position_ids.flatten()
        # cos = cos[f_position_ids]
        # sin = sin[f_position_ids]
        cos = torch.index_select(cos, 0, f_position_ids)
        sin = torch.index_select(sin, 0, f_position_ids)
        if kv_layout == "BNSD":
            dim = 1
        else:
            dim = 2
        cos = cos.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(dim)  # [bs, 1, seq_len, dim]
        sin = sin.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(dim)  # [bs, 1, seq_len, dim]

    b, s, n, d = q.shape
    if n > 32:
        q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
        k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    else:
        q_embed, k_embed = torch_npu.npu_apply_rotary_pos_emb(q, k, cos, sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device_num = int(os.getenv("DEVICE_NUM", 1))
        self.ffn_mm_merge = os.getenv("FFN_MM_MERGE", "False")
        self.convert_ffn_merge = os.getenv("CONVERT_FFN_MERGE", "False")
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        if self.convert_ffn_merge == "True":
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.merge_up_gate = nn.Linear(self.hidden_size, 2*self.intermediate_size, bias=False)
        else:
            if self.ffn_mm_merge == "True":
                self.merge_up_gate = nn.Linear(self.hidden_size, 2*self.intermediate_size, bias=False)
            else:
                self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
                self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]
        self.input_scale = torch.ones(
            self.hidden_size, dtype=torch.float32).npu()
        self.input_offset = torch.ones(
            self.hidden_size, dtype=torch.int32).npu()

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i])
                                  for i in range(self.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i])
                                for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(
                gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i])
                         for i in range(self.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            if self.ffn_mm_merge == "True":
                mixed_x = self.merge_up_gate(x)
                up_state, gate_state = mixed_x.split(self.intermediate_size // self.device_num, dim=-1)
                down_proj = self.down_proj(self.act_fn(gate_state) * up_state)
            else:
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale_value = 1 / math.sqrt(self.head_dim)
        self.device_num = int(os.getenv("DEVICE_NUM", 1))
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings
        self.q_hidden_size = self.num_heads * self.head_dim // self.device_num
        self.kv_hidden_size = self.num_key_value_heads * self.head_dim // self.device_num
        self.soc_version = os.getenv("SOC_VERSION", "Ascend910B2")
        self.mm_merge = os.getenv("MM_MERGE", "False")
        self.kv_layout = os.getenv("KV_LAYOUT", "BSND")
        input_dtype = os.getenv("DTYPE", "fp16")
        if input_dtype == "fp16":
            self.dtype = torch.float16
        elif input_dtype == "fp32":
            self.dtype = torch.float32
        elif input_dtype == "bf16":
            self.dtype = torch.bfloat16
        else:
            raise TypeError("DTYPE is not defined in model config")
        quant_type = os.getenv("QUANT_TYPE", None)
        self.scale = torch.ones(
            self.head_dim, device='npu', dtype=torch.float32)
        self.offset = torch.ones(
            self.head_dim, device='npu', dtype=torch.int32)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False)
        if self.mm_merge == "True":
            self.qkv = nn.Linear(self.hidden_size, self.num_heads * self.head_dim +
                                 2 * self.num_key_value_heads * self.head_dim, bias=False)
        # self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_len: Optional[list] = None,
            kv_padding_size: Optional[torch.LongTensor] = None,
            rotary_emb_cos: Optional[torch.Tensor] = None,
            rotary_emb_sin: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads *
                                 self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i])
                            for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i])
                          for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i])
                            for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            if self.mm_merge == "True":
                mixed_x_layer = self.qkv(hidden_states)
                query_states, key_states, value_states = mixed_x_layer.split(
                    [self.q_hidden_size, self.kv_hidden_size, self.kv_hidden_size], dim=2)
            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, rotary_emb_cos.to(value_states.dtype),
                                                        rotary_emb_sin.to(value_states.dtype))

        if self.kv_layout == "BNSD":
            if q_len == 1 or self.num_heads == 1:
                query_states = query_states.view(bsz, self.num_heads, q_len, self.head_dim)
            else:
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            if q_len == 1 or self.num_key_value_heads == 1:
                key_states = key_states.view(bsz, self.num_key_value_heads, q_len, self.head_dim)
                value_states = value_states.view(bsz, self.num_key_value_heads, q_len, self.head_dim)
            else:
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        tmp_key = past_key_value[0]
        tmp_value = past_key_value[1]
        if self.kv_layout == "BNSD":
            axis = -2
        else:
            axis = 1

        if q_len > 1:
            tmp_ids = torch.zeros(bsz, dtype=torch.int64,
                                  device=position_ids.device)
            torch_npu.scatter_update_(tmp_key, tmp_ids, key_states, axis)
            torch_npu.scatter_update_(tmp_value, tmp_ids, value_states, axis)
        elif q_len == 1:
            torch_npu.scatter_update_(
                tmp_key, position_ids.reshape(-1), key_states, axis)
            torch_npu.scatter_update_(
                tmp_value, position_ids.reshape(-1), value_states, axis)
        past_key = tmp_key
        past_value = tmp_value
        past_key_value = (past_key, past_value) if use_cache else None

        if q_len > 1:
            attention_mask_pfa = attention_mask.to(torch.bool)
            attn_output = torch_npu.npu_prompt_flash_attention(query_states, key_states.contiguous(),
                                                               value_states.contiguous(), num_heads=self.num_heads,
                                                               input_layout=self.kv_layout,
                                                               scale_value=self.scale_value,
                                                               pre_tokens=65535, next_tokens=0,
                                                               atten_mask=attention_mask_pfa,
                                                               num_key_value_heads=self.num_key_value_heads)
        else:
            attention_mask_ifa = attention_mask.to(torch.bool)
            attn_output = torch_npu.npu_incre_flash_attention(query_states, past_key.contiguous(),
                                                              past_value.contiguous(), num_heads=self.num_heads,
                                                              input_layout=self.kv_layout,
                                                              scale_value=self.scale_value,
                                                              atten_mask=attention_mask_ifa,
                                                              actual_seq_lengths=kv_len,
                                                              kv_padding_size=kv_padding_size,
                                                              num_key_value_heads=self.num_key_value_heads)
        if self.kv_layout == "BNSD":
            batch, head_num, slen, head_dim = attn_output.shape
            if head_num == 1 or slen == 1:
                attn_output = attn_output.reshape(batch, slen, head_num, head_dim)
            else:
                attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i])
                              for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # add + rmsnorm 融合
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            past_residual: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_len: Optional[list] = None,
            kv_padding_size: Optional[torch.LongTensor] = None,
            rotary_emb_cos: Optional[torch.Tensor] = None,
            rotary_emb_sin: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        hidden_states, residual = self.input_layernorm(
            hidden_states, past_residual)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_len=kv_len,
            kv_padding_size=kv_padding_size,
            rotary_emb_cos=rotary_emb_cos,
            rotary_emb_sin=rotary_emb_sin,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # hidden_states = residual + hidden_states

        # Fully Connected
        # residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        # hidden_states = residual + hidden_states

        outputs = (residual, hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.num_attention_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.torch_dtype = config.torch_dtype
        self.num_key_value_heads = config.num_key_value_heads

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self._init_rope()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _prepare_decoder_rotray_cos_sin(self, position_ids):
        cos, sin = self.rotary_emb()
        # import torch_npu
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        # with torch.autograd.profiler.record_function("position_ids1111111"):
        f_position_ids = position_ids.flatten()
        # cos = cos[f_position_ids]
        # sin = sin[f_position_ids]
        cos = torch.index_select(cos, 0, f_position_ids)
        sin = torch.index_select(sin, 0, f_position_ids)
        cos = cos.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)  # [bs, 1, seq_len, dim]
        sin = sin.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)  # [bs, 1, seq_len, dim]
        return (cos, sin)

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

        # b, s = input_shape
        # if s > 1:
        #     combined_attention_mask = torch.triu(torch.ones((1, 1, 2048, 2048), dtype=torch.bool, device=inputs_embeds.device), diagonal=1)
        # else:
        #     combined_attention_mask = None
        #     if input_shape[-1] > 1:
        #         combined_attention_mask_first = _make_causal_mask(
        #             input_shape,
        #             inputs_embeds.dtype,
        #             device=inputs_embeds.device,
        #             past_key_values_length=past_key_values_length,
        #         )
        #         combined_attention_mask = torch.zeros(input_shape[0], 1, input_shape[1], past_key_values_length,
        #                                               dtype=torch.bool, device=inputs_embeds.device)
        #         combined_attention_mask[:, :, :, 0:input_shape[1]] = combined_attention_mask_first
        #     if attention_mask is not None:
        #         attention_mask = attention_mask.to(torch.bool)
        #         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #         expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
        #             inputs_embeds.device
        #         )
        #         # combined_attention_mask = (
        #         #     expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        #         # )
        #         combined_attention_mask = (
        #             expanded_attn_mask if combined_attention_mask is None else torch.logical_or(expanded_attn_mask, combined_attention_mask)
        #         )

        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask_first = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
            combined_attention_mask = torch.zeros(input_shape[0], 1, input_shape[1], past_key_values_length,
                                                  dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            combined_attention_mask[:, :, :, 0:input_shape[1]
                                    ] = combined_attention_mask_first
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                combined_attention_mask
            )

        return combined_attention_mask.to(torch.bool)

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            kv_len: Optional[list] = None,
            kv_padding_size: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(2, -1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )

        hidden_states = inputs_embeds
        residual = None

        rotary_emb_cos, rotary_emb_sin = self._prepare_decoder_rotray_cos_sin(
            position_ids[0])
        model_position_ids = position_ids[1].to(torch.int32)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    past_residual=residual,
                    attention_mask=attention_mask,
                    position_ids=model_position_ids,
                    kv_len=kv_len,
                    kv_padding_size=kv_padding_size,
                    rotary_emb_cos=rotary_emb_cos,
                    rotary_emb_sin=rotary_emb_sin,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            residual = layer_outputs[0]
            hidden_states = layer_outputs[1]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[3 if output_attentions else 2],)

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        # hidden_states = hidden_states + residual
        hidden_states, _ = self.norm(hidden_states, residual)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, world_size=1):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        self.world_size = world_size
        # Initialize weights and apply final processing
        self.post_init()
        self.run_id = 0
        self.device_num = int(os.getenv("DEVICE_NUM", 1))
        self.quant_type = os.getenv("QUANT_TYPE", None)
        self.actual_seq = os.getenv("ACTUAL_SEQ", "False")
        self.kv_quant = os.getenv("KV_QUANT", "true")
        self.kv_layout = os.getenv("KV_LAYOUT", "BSND")

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def merge_qkv_weight(self, model, world_size):
        assert model is not None
        assert world_size >= 1

        def to_parameter(data):
            return nn.Parameter(data, requires_grad=False)

        # [out_channel, in_channel]
        qw_size = self.model.layers[0].self_attn.q_proj.weight.shape
        # [out_channel, in_channel]
        kw_size = self.model.layers[0].self_attn.k_proj.weight.shape
        # [out_channel, in_channel]
        vw_size = self.model.layers[0].self_attn.v_proj.weight.shape

        qstep_size = qw_size[0] // world_size
        kstep_size = kw_size[0] // world_size
        vstep_size = vw_size[0] // world_size
        print(f"q k v out channel size:{qstep_size} {kstep_size} {vstep_size}")

        for i, block in enumerate(model.model.layers):
            qw = self.model.layers[i].self_attn.q_proj.weight
            kw = self.model.layers[i].self_attn.k_proj.weight
            vw = self.model.layers[i].self_attn.v_proj.weight
            qkvw = self.model.layers[i].self_attn.qkv.weight

            weight_list = []
            for j in range(world_size):
                curr_qw = qw[j * qstep_size:(j + 1) * qstep_size, :]
                curr_kv = kw[j * kstep_size:(j + 1) * kstep_size, :]
                curr_vw = vw[j * vstep_size:(j + 1) * vstep_size, :]
                if i == 0:
                    print(curr_qw.shape, curr_kv.shape, curr_vw.shape)
                weight_list.append(to_parameter(
                    torch.cat([curr_qw, curr_kv, curr_vw], axis=0)))

            if len(weight_list) == 1:
                self.model.layers[i].self_attn.qkv.weight = weight_list[0]
            else:
                self.model.layers[i].self_attn.qkv.weight = to_parameter(
                    torch.cat(weight_list, axis=0))
            if i == 0:
                print(
                    f"world_size:{world_size} len weight_list:{len(weight_list)}")
                print(
                    f"qkv weight shape:{self.model.layers[i].self_attn.qkv.weight.shape}")

    def merge_up_gate_weight(self, model, world_size):
        assert model is not None
        assert world_size >= 1
        print(f"enter split_up_gate_weight")

        def to_parameter(data):
            return nn.Parameter(data, requires_grad=False)

        uw_size = self.model.layers[0].mlp.up_proj.weight.shape    # [out_channel, in_channel]
        gw_size = self.model.layers[0].mlp.gate_proj.weight.shape    # [out_channel, in_channel]

        print(f"uw_size:{uw_size} gw_size:{gw_size}")

        ustep_size = uw_size[0] // world_size
        gstep_size = gw_size[0] // world_size

        print(f"ustep_size:{ustep_size} gstep_size:{gstep_size}")

        for i, block in enumerate(model.model.layers):
            uw = self.model.layers[i].mlp.up_proj.weight
            gw = self.model.layers[i].mlp.gate_proj.weight
            merge_up_gate = self.model.layers[i].mlp.merge_up_gate.weight

            weight_list=[]
            for j in range(world_size):
                curr_uw=uw[j * ustep_size:(j+1) * ustep_size, :]
                curr_gw=gw[j * gstep_size:(j+1) * gstep_size, :]
                if i == 0:
                    print(curr_uw.shape, curr_gw.shape)
                weight_list.append(to_parameter(torch.cat([curr_uw, curr_gw], axis=0)))

            if len(weight_list) == 1:
                self.model.layers[i].mlp.merge_up_gate.weight = weight_list[0]
            else:
                self.model.layers[i].mlp.merge_up_gate.weight = to_parameter(torch.cat(weight_list, axis=0))
            if i == 0:
                print(f"world_size:{world_size} len weight_list:{len(weight_list)}")
                print(f"merge_up_gate weight shape:{self.model.layers[i].mlp.merge_up_gate.weight.shape}")

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            kv_len: Optional[list] = None,
            kv_padding_size: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            kv_len=kv_len,
            kv_padding_size=kv_padding_size,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i])
                      for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            bs, seq_in, hidden = hidden_states.size()
            if seq_in > 1:
                gather_index = torch.ones(
                    bs, dtype=torch.int64, device='npu') * (seq_in - 1)
                gather_index = gather_index.unsqueeze(
                    dim=1).unsqueeze(dim=2).repeat(1, 1, hidden)
                hidden_states = torch.gather(hidden_states, 1, gather_index)
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        input_ids = input_ids.contiguous().clone()

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        batch_size, seq_length = input_ids.shape
        self.run_id = 0 if seq_length > 1 else self.run_id

        # 1.generate 1st past_key_values
        if past_key_values is None:
            dtype = os.getenv("DTYPE", "fp16")
            if dtype == "fp16":
                kv_chahe_dtype = torch.float16
            elif dtype == "fp32":
                kv_chahe_dtype = torch.float32
            elif dtype == "bf16":
                kv_chahe_dtype = torch.bfloat16
            else:
                raise TypeError("DTYPE is not defined in model config")
            if self.quant_type is not None:
                if self.kv_quant == 'true':
                    kv_chahe_dtype = torch.int8

            past_key_values = ()
            for i in range(self.model.num_hidden_layers):
                # shape: [1, num_attention_heads, 2048, 128] (32, 52, 2048, 128) 60层
                if self.kv_layout == "BSND":
                    kv_shape = (
                        batch_size, self.model.max_position_embeddings, self.model.num_key_value_heads // self.device_num,
                        self.model.hidden_size // self.model.num_attention_heads)
                else:
                    kv_shape = (
                        batch_size, self.model.num_key_value_heads // self.device_num, self.model.max_position_embeddings,
                        self.model.hidden_size // self.model.num_attention_heads)
                k_cache = torch.zeros(
                    kv_shape, dtype=kv_chahe_dtype, device=input_ids.device)
                v_cache = torch.zeros(
                    kv_shape, dtype=kv_chahe_dtype, device=input_ids.device)
                past_key_values += ((k_cache, v_cache),)

        # 2.modify position_ids
        if seq_length > 1:
            position_ids = torch.stack((position_ids, position_ids))
            self.mask_id = seq_length
            self.padding_mask = torch.zeros(
                batch_size, self.model.max_position_embeddings, device=position_ids.device)
        else:
            mask_position = torch.ones(position_ids.shape, dtype=position_ids.dtype, device=position_ids.device) * (
                self.mask_id + self.run_id - 1)
            position_ids = torch.stack((position_ids, mask_position))
        kv_padding_size = torch.tensor(self.model.max_position_embeddings - self.mask_id - self.run_id,
                                       device=position_ids.device)
        # 3.padding attention_mask
        padding_mask = self.padding_mask
        padding_mask[:, :attention_mask.shape[1]] = attention_mask
        attention_mask = self.model._prepare_decoder_attention_mask(
            padding_mask, (batch_size,
                           seq_length), past_key_values[0][0], self.model.max_position_embeddings,
        )

        if seq_length > 1:
            attention_mask = attention_mask[..., :seq_length]

        self.run_id += 1
        kv_len = (position_ids[0][:, -1] + 1).cpu().tolist()

        if self.actual_seq == "True":
            torch._dynamo.mark_static(input_ids)
            torch._dynamo.mark_static(attention_mask)
            torch._dynamo.mark_static(position_ids)
            torch._dynamo.mark_static(kv_padding_size)
            for i in range(self.model.num_hidden_layers):
                torch._dynamo.mark_static(past_key_values[i][0])
                torch._dynamo.mark_static(past_key_values[i][1])
        else:
            kv_padding_size = None
            kv_len = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "kv_len": kv_len,
                "kv_padding_size": kv_padding_size,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                      for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(
            batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )