# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import math
import inspect
from importlib import import_module
from typing import Callable, List, Optional, Union

import torch
import torch_npu
import torch.nn.functional as F
from torch import nn
from diffusers.models.attention_processor import Attention

import torchair._contrib.custom_torch_ops


class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        unnormed_hidden_states: torch.FloatTensor = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, \
                as passing it will raise an error in the future. `scale` should directly be passed while calling \
                the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be 
            # batch, heads, source_length, target_length
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # the output of npu_pfa = (batch, num_heads*seq_len, head_dim)
        hidden_states = torch.ops.npu.npu_prompt_flash_attention(
            query.contiguous(), key.contiguous(), value.contiguous(),
            num_heads=attn.heads, input_layout="BSH", atten_mask=attention_mask,
            scale_value=(1 / math.sqrt(head_dim)),
            sparse_mode=10)

        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor2_0Tome:
    def __init__(self, h=64, w=64, sy=2, sx=2):
        hsy, wsx = h // sy, w // sx
        self.num_dst = hsy * wsx

        generator = torch.Generator()
        # For each sy * sx kernel, randomly assign one token to be dst and the rest src
        rand_idx = torch.zeros(hsy, wsx, 1, dtype=torch.int64)

        # Ensure image divided by sy and sx
        idx_buffer_view = torch.zeros(hsy, wsx, sy * sx, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        del idx_buffer, idx_buffer_view

        self.a_idx = rand_idx[:, self.num_dst:, :].cpu().numpy().tolist()  # tome src tokens
        self.b_idx = rand_idx[:, :self.num_dst, :].cpu().numpy().tolist()  # tome dst tokens

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            unnormed_hidden_states: torch.FloatTensor = None,
        ) -> torch.FloatTensor:
        residual = hidden_states

        args = ()
       
        bs, seqlen, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        
        # Step 1: calc similarities before norm
        a_idx = torch.tensor(self.a_idx).to(hidden_states.device)
        b_idx = torch.tensor(self.b_idx).to(hidden_states.device)

        unnormed_hidden_states = unnormed_hidden_states / unnormed_hidden_states.norm(dim=-1, keepdim=True)

        a = torch.index_select(unnormed_hidden_states, dim=1, index=a_idx.reshape(seqlen - self.num_dst))
        b = torch.index_select(unnormed_hidden_states, dim=1, index=b_idx.reshape(self.num_dst))

        scores = a @ b.transpose(-1, -2)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        
        src = torch.index_select(hidden_states, dim=1, index=a_idx.reshape(seqlen - self.num_dst))
        dst = torch.index_select(hidden_states, dim=1, index=b_idx.reshape(self.num_dst))

        # Step 2: to token merge
        unm, unreduce_token, unreduced_count = torch.ops.npu_inference.npu_tome_merge(
            src, dst, edge_idx[..., 0], node_idx, top_rate=0.5)
        merged = torch.sum(unreduce_token, dim=1) / torch.unsqueeze(torch.sum(unreduced_count, dim=1), 2)
        hidden_states = torch.cat([unm, merged], dim=1).to(torch.float16).to(hidden_states.device)

        query = attn.to_q(hidden_states, *args)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        hidden_states = torch.ops.npu.npu_prompt_flash_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            num_heads=attn.heads,
            input_layout="BSH",
            atten_mask=attention_mask,
            scale_value=1. / math.sqrt(head_dim),
            sparse_mode=10
        ).to(query.dtype)

        # linear projection
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # Step 3: token unmerge
        hidden_states = torch.ops.npu_inference.npu_tome_unmerge(
            hidden_states,
            a_idx.reshape(1, seqlen - self.num_dst).expand(bs, -1).contiguous(),
            b_idx.reshape(1, self.num_dst).expand(bs, -1).contiguous(),
            edge_idx[..., 0], node_idx, top_rate=0.5
        )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states