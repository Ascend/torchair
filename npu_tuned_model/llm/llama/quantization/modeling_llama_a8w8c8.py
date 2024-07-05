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
from quantization.npu_quantize import NpuA8W8Linear

from torch.nn import Parameter
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, 
    CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, 
    add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.llama.configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


# adapter __make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
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

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), 1).to(dtype)


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
        if residual is None:
            return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0], hidden_states
        else:
            y, _, x = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
        return y, x


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


    def __forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

    def forward(self, x=None, seq_len=None):
        if x is None and seq_len is None:
            return self.cos_cached, self.sin_cached

        return (
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
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


def apply_rotary_pos_emb(q, k, cos, sin):
    return torch_npu.npu_apply_rotary_pos_emb(q, k, cos, sin)


class LlamaMLP(nn.Module):
    # FP16 原__init__函数
    def ____init__(self, config):
        super().__init__()
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def __init__(self, config):
        super().__init__()
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = NpuA8W8Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = NpuA8W8Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = NpuA8W8Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]

        self.register_buffer("scale_up", torch.ones(self.hidden_size, dtype=torch.float32, device="npu"))
        self.register_buffer("offset_up", torch.ones(self.hidden_size, dtype=torch.int32, device="npu"))
        self.register_buffer("scale_down", torch.ones(self.intermediate_size, dtype=torch.float32, device="npu"))
        self.register_buffer("offset_down", torch.ones(self.intermediate_size, dtype=torch.int32, device="npu"))

    # FP16 原 forward 函数
    def _forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

    def forward(self, x):
        x = torch_npu.npu_quantize(x, self.scale_up, self.offset_up, torch.qint8, axis=-1)
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        x = torch_npu.npu_quantize(x, self.scale_down, self.offset_down, torch.qint8, axis=-1)
        return self.down_proj(x)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # FP16 原__init__函数
    def ___init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        self.scale_value = 1 / math.sqrt(self.head_dim)
        # q k v merge config
        self.q_hidden_size = self.num_heads * self.head_dim // config.world_size
        self.kv_hidden_size = self.num_key_value_heads * self.head_dim // config.world_size

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.qkv = nn.Linear(self.hidden_size,
                             self.num_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim, bias=False)

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        self.scale_value = 1 / math.sqrt(self.head_dim)
        # q k v merge config
        self.q_hidden_size = self.num_heads * self.head_dim // config.world_size
        self.kv_hidden_size = self.num_key_value_heads * self.head_dim // config.world_size

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.o_proj = NpuA8W8Linear(self.q_hidden_size, self.hidden_size)
        self.qkv = nn.Linear(self.hidden_size, self.q_hidden_size + 2 * self.kv_hidden_size)

        ## qkvo activation asymmetric quantization parameters
        self.register_buffer("scale_qkv", torch.ones(self.hidden_size, dtype=torch.float32, device="npu"))
        self.register_buffer("offset_qkv", torch.ones(self.hidden_size, dtype=torch.int32, device="npu"))
        self.register_buffer("scale_o", torch.ones(self.q_hidden_size, dtype=torch.float32, device="npu"))
        self.register_buffer("offset_o", torch.ones(self.q_hidden_size, dtype=torch.int32, device="npu"))
        
        ## kvcache int8 asymmetric quantization parameters
        self.register_buffer("kcache_scale", torch.ones(self.kv_hidden_size, dtype=torch.float32, device="npu"))
        self.register_buffer("kcache_offset", torch.ones(self.kv_hidden_size, dtype=torch.int32, device="npu"))
        self.register_buffer("vcache_scale", torch.ones(self.kv_hidden_size, dtype=torch.float32, device="npu"))
        self.register_buffer("vcache_offset", torch.ones(self.kv_hidden_size, dtype=torch.int32, device="npu"))
        self.register_buffer("ifa_antiquant_scale", torch.ones((2, self.kv_hidden_size), 
                                                    dtype=torch.float16, device="npu"))
        self.register_buffer("ifa_antiquant_offset", torch.ones((2, self.kv_hidden_size), 
                                                    dtype=torch.float16, device="npu"))

    # FP16 原 forward 函数
    def _forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        updated_kv_positions: Optional[torch.LongTensor] = None,
        kv_padding_size: Optional[torch.LongTensor] = None,
        actual_seq_len: Optional[list] = None,
        rotary_emb_cos: Optional[torch.Tensor] = None,
        rotary_emb_sin: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        # qkv融合后参与矩阵乘计算，然后将计算结果进行拆分
        qkv_states = self.qkv(hidden_states)
        query_states, key_states, value_states = qkv_states.split(
            [self.q_hidden_size, self.kv_hidden_size, self.kv_hidden_size], dim=2)

        # format BSND
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        rotary_emb_cos.to(value_states.dtype),
                                                        rotary_emb_sin.to(value_states.dtype))

        # 更新指定位置上的kv cache，position_ids在全量图执行时从seq_len 0的位置更新，在增量图执行时从seq_len位置更新
        tmp_ids = updated_kv_positions.reshape(-1)
        # format BSND, 1 means seq_len dim index
        torch_npu.scatter_update_(past_key_value[0], tmp_ids, key_states, 1)
        torch_npu.scatter_update_(past_key_value[1], tmp_ids, value_states, 1)

        key_states1 = past_key_value[0] if q_len == 1 else key_states
        value_states1 = past_key_value[1] if q_len == 1 else value_states

        past_key_value = past_key_value if use_cache else None

        attention_mask = attention_mask.to(torch.bool)
        if q_len > 1:
            attn_output = torch_npu.npu_prompt_flash_attention(query_states, key_states1.contiguous(),
                                                               value_states1.contiguous(), num_heads=self.num_heads,
                                                               input_layout="BSND",
                                                               scale_value=self.scale_value,
                                                               pre_tokens=65535, next_tokens=0,
                                                               atten_mask=attention_mask,
                                                               num_key_value_heads=self.num_key_value_heads)
        else:
            attn_output = torch_npu.npu_incre_flash_attention(query_states, key_states1.contiguous(),
                                                              value_states1.contiguous(), num_heads=self.num_heads,
                                                              input_layout="BSND",
                                                              scale_value=self.scale_value,
                                                              atten_mask=attention_mask,
                                                              actual_seq_lengths=actual_seq_len,
                                                              kv_padding_size=kv_padding_size,
                                                              num_key_value_heads=self.num_key_value_heads)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        updated_kv_positions: Optional[torch.LongTensor] = None,
        kv_padding_size: Optional[torch.LongTensor] = None,
        actual_seq_len: Optional[list] = None,
        rotary_emb_cos: Optional[torch.Tensor] = None,
        rotary_emb_sin: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        # 对activation做量化，fp16->int8
        hidden_states = torch_npu.npu_quantize(hidden_states, self.scale_qkv, self.offset_qkv, torch.qint8, axis=-1)

        # qkv融合后参与矩阵乘计算，然后将计算结果进行拆分
        qkv_states = self.qkv(hidden_states)
        query_states, key_states, value_states = qkv_states.split(
            [self.q_hidden_size, self.kv_hidden_size, self.kv_hidden_size], dim=2)
        # format BSND
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        rotary_emb_cos.to(value_states.dtype),
                                                        rotary_emb_sin.to(value_states.dtype))

        # kvcache int8 perchannel, asymmetric
        rshp_key = key_states.reshape(bsz, q_len, self.kv_hidden_size)
        quant_key = torch_npu.npu_quantize(rshp_key, self.kcache_scale, self.kcache_offset, torch.qint8, axis=-1)
        quant_key = quant_key.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        rsh_value = value_states.reshape(bsz, q_len, self.kv_hidden_size)
        quant_value = torch_npu.npu_quantize(rsh_value, self.vcache_scale, self.vcache_offset, torch.qint8, axis=-1)
        quant_value = quant_value.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # 更新指定位置上的kv cache，position_ids在全量图执行时从seq_len 0的位置更新，在增量图执行时从seq_len位置更新
        tmp_ids = updated_kv_positions.reshape(-1)
        # format BSND, 1 means seq_len dim index
        torch_npu.scatter_update_(past_key_value[0], tmp_ids, quant_key, 1)
        torch_npu.scatter_update_(past_key_value[1], tmp_ids, quant_value, 1)

        key_states1 = past_key_value[0] if q_len == 1 else key_states
        value_states1 = past_key_value[1] if q_len == 1 else value_states

        past_key_value = past_key_value if use_cache else None

        attention_mask = attention_mask.to(torch.bool)
        if q_len > 1:
            attn_output = torch_npu.npu_prompt_flash_attention(query_states, key_states1.contiguous(),
                                                               value_states1.contiguous(), num_heads=self.num_heads,
                                                               input_layout="BSND",
                                                               scale_value=self.scale_value,
                                                               pre_tokens=65535, next_tokens=0,
                                                               atten_mask=attention_mask,
                                                               num_key_value_heads=self.num_key_value_heads)
        else:
            attn_output = torch_npu.npu_incre_flash_attention(query_states, key_states1.contiguous(),
                                                              value_states1.contiguous(), num_heads=self.num_heads,
                                                              input_layout="BSND",
                                                              scale_value=self.scale_value,
                                                              antiquant_scale=self.ifa_antiquant_scale,
                                                              antiquant_offset=self.ifa_antiquant_offset,
                                                              atten_mask=attention_mask,
                                                              actual_seq_lengths=actual_seq_len,
                                                              kv_padding_size=kv_padding_size,
                                                              num_key_value_heads=self.num_key_value_heads)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = torch_npu.npu_quantize(attn_output, self.scale_o, self.offset_o, torch.qint8, axis=-1)
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
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        updated_kv_positions: Optional[torch.LongTensor] = None,
        kv_padding_size: Optional[torch.LongTensor] = None,
        actual_seq_len: Optional[list] = None,
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

        hidden_states, residual = self.input_layernorm(hidden_states, past_residual)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            updated_kv_positions=updated_kv_positions,
            kv_padding_size=kv_padding_size,
            actual_seq_len=actual_seq_len,
            rotary_emb_cos=rotary_emb_cos,
            rotary_emb_sin=rotary_emb_sin,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

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

    This model is also a PyTorch [torch.nn.Module] subclass.
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

    # FP16 原 _init_weights 函数
    def __init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, NpuA8W8Linear):
            module.weight.data.zero_()
            module.bias.data.zero_()
            module.deq_scale.data.zero_()
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
            and modify to your needs. See diagram 1 in [the paper] for more
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

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        #add config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.torch_dtype = config.torch_dtype

        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self._init_rope()

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # add new func
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
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

    # add new func
    def _prepare_decoder_rotary_cos_sin(self, position_ids):
        cos, sin = self.rotary_emb()
        cos = cos.squeeze(1).squeeze(0)
        sin = sin.squeeze(1).squeeze(0)
        f_position_ids = position_ids.flatten()
        cos = torch.index_select(cos, 0, f_position_ids)
        sin = torch.index_select(sin, 0, f_position_ids)
        cos = cos.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)
        sin = sin.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)
        return cos, sin

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        updated_kv_positions: Optional[torch.LongTensor] = None,
        kv_padding_size: Optional[torch.LongTensor] = None,
        actual_seq_len: Optional[list] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
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
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0

        if seq_length == 1:
            past_key_values_length = self.max_position_embeddings

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # 此处处理cos， sin是为了不在每层里面重复计算
        rotary_emb_cos, rotary_emb_sin = self._prepare_decoder_rotary_cos_sin(position_ids)

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

        residual = None
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
                    updated_kv_positions=updated_kv_positions,
                    kv_padding_size=kv_padding_size,
                    actual_seq_len=actual_seq_len,
                    rotary_emb_cos=rotary_emb_cos,
                    rotary_emb_sin=rotary_emb_sin,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            residual = layer_outputs[0]
            hidden_states = layer_outputs[1]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 2],)

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

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

    def __init__(self, config):
        super().__init__(config)
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        config.world_size = self.world_size
        self.model = LlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.prompt_length = None
        self.updated_kv_positions = None

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

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        updated_kv_positions: Optional[torch.LongTensor] = None,
        kv_padding_size: Optional[torch.LongTensor] = None,
        actual_seq_len: Optional[list] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
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
            updated_kv_positions=updated_kv_positions,
            kv_padding_size=kv_padding_size,
            actual_seq_len=actual_seq_len,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # 由于logits最后也只取[:,-1,:]，相当于只取最新seq位置上的数据，l
            # 所以在全量的最后线性层计算可以只对最新的seq位置做计算，降低计算量
            bs, seq, hidden = hidden_states.size()
            if seq > 1:
                gather_index = torch.ones(bs, dtype=torch.int64, device=hidden_states.device) * (seq - 1)
                gather_index = gather_index.unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, 1, hidden)
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
        self.prompt_length = input_ids.shape[1]
        if past_key_values:
            input_ids = input_ids[:, -1:]
        input_ids = input_ids.clone()  # 添加clone目的是为了保证fx图上input_ids不变化
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

        # 固定kv cache的大小，用作全量图和增量图的kv cache更新
        batch_size, seq_length = input_ids.shape
        if past_key_values is None:
            kv_shape = (
                batch_size, self.model.max_position_embeddings, self.model.num_key_value_heads // self.world_size,
                self.model.hidden_size // self.model.num_attention_heads)
            if kwargs.get("kv_tensors", None) is None:
                past_key_values = ()
                for i in range(self.model.num_hidden_layers):
                    k_cache = torch.zeros(kv_shape, dtype=torch.int8, device=input_ids.device)
                    v_cache = torch.zeros(kv_shape, dtype=torch.int8, device=input_ids.device)
                    past_key_values += ((k_cache, v_cache),)
            else:
                past_key_values = kwargs["kv_tensors"]

        # 将attention mask的创建挪到最外层主要是为了在图模式场景下固定输入
        # 增量图attention_mask padding到最大长度
        # 增加updated_kv_positions给固定kv cache的tensor更新提供更新位置
        past_key_values_length = 0
        if seq_length > 1:
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=input_ids.device)
            self.updated_kv_positions = torch.zeros(batch_size, dtype=position_ids.dtype, device=position_ids.device)
        else:
            bsz, src_len = attention_mask.size()
            padding_mask = torch.zeros(batch_size, self.model.max_position_embeddings, device=input_ids.device)
            padding_mask[:, :src_len] = attention_mask
            attention_mask = padding_mask
            past_key_values_length = self.model.max_position_embeddings
            self.updated_kv_positions = torch.ones(position_ids.shape, dtype=position_ids.dtype,
                                                   device=position_ids.device) * (self.prompt_length - 1)

        attention_mask = self.model._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), past_key_values[0][0], past_key_values_length
        )
        # ifa Computational optimization inputs
        kv_padding_size = torch.tensor(self.model.max_position_embeddings - self.prompt_length,
                                       device=position_ids.device)
        actual_seq_len = (position_ids[:, -1] + 1).cpu().tolist()

        position_ids = position_ids.clone() # 添加clone目的是为了保证fx图上position_ids不变化
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "updated_kv_positions": self.updated_kv_positions,
                "kv_padding_size": kv_padding_size,
                "actual_seq_len": actual_seq_len,
            }
        )

        # 在走动态图模式场景下，实际增量是静态图，所以标记增量的输入tensor为static
        # fa算子设置了actual_seq_len，需要走动态图mark_static
        if actual_seq_len is not None:
            self._mark_model_inputs_static(model_inputs)

        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    # add new func
    def merge_qkv_weight(self, tp_size=1):
        if self.model is None:
            raise ValueError("Model is None, please check")

        def _to_parameter(data):
            return nn.Parameter(data, requires_grad=False)

        qw_size = self.model.layers[0].self_attn.q_proj.weight.shape  # [out_channel, in_channel]
        kw_size = self.model.layers[0].self_attn.k_proj.weight.shape
        vw_size = self.model.layers[0].self_attn.v_proj.weight.shape

        q_sliced_size = qw_size[0] // tp_size
        k_sliced_size = kw_size[0] // tp_size
        v_sliced_size = vw_size[0] // tp_size
        print(f"sliced out channel size, q:{q_sliced_size}, k:{k_sliced_size}, v:{v_sliced_size}")

        for i in range(len(self.model.layers)):
            qw = self.model.layers[i].self_attn.q_proj.weight
            kw = self.model.layers[i].self_attn.k_proj.weight
            vw = self.model.layers[i].self_attn.v_proj.weight

            weight_list = []
            for j in range(tp_size):
                sliced_qw = qw[j * q_sliced_size: (j + 1) * q_sliced_size, :]
                sliced_kw = kw[j * k_sliced_size: (j + 1) * k_sliced_size, :]
                sliced_vw = vw[j * v_sliced_size: (j + 1) * v_sliced_size, :]
                weight_list.append(_to_parameter(torch.cat([sliced_qw, sliced_kw, sliced_vw], axis=0)))

            if len(weight_list) == 1:
                self.model.layers[i].self_attn.qkv.weight = weight_list[0]
            else:
                self.model.layers[i].self_attn.qkv.weight = _to_parameter(torch.cat(weight_list, axis=0))

    def _mark_model_inputs_static(self, model_inputs):
        for key, value in model_inputs.items():
            if key == "past_key_values":
                for i in range(self.model.num_hidden_layers):
                    torch._dynamo.mark_static(value[i][0])
                    torch._dynamo.mark_static(value[i][1])
            elif isinstance(value, torch.Tensor):
                torch._dynamo.mark_static(value)