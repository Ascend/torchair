# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
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

import os
import argparse
import logging
import shutil
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from models.modeling_qwen3_moe import Qwen3MoeForCausalLM
from models.modeling_qwen3_moe_ori import Qwen3MoeForCausalLMOri

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)


def _to_parameter(data):
    return nn.Parameter(data, requires_grad=False)


def split_w_moe(block, dst_model, i, local_rank):
    dst_model.model.layers[i].mlp.gate.weight.data = block.mlp.gate.weight.data
    expert_num = dst_model.config.num_experts
    gate_proj_list, down_proj_list, up_proj_list = [], [], []  
    for j, src_expert in enumerate(block.mlp.experts):
        ffn_dim = dst_model.model.layers[i].mlp.experts.intermediate_size_per_rank
        gate_proj_list.append(
            src_expert.gate_proj.weight.data[local_rank * ffn_dim: (local_rank + 1) * ffn_dim, :].contiguous())
        up_proj_list.append(
            src_expert.up_proj.weight.data[local_rank * ffn_dim: (local_rank + 1) * ffn_dim, :].contiguous())
        down_proj_list.append(
            src_expert.down_proj.weight.data[:, local_rank * ffn_dim: (local_rank + 1) * ffn_dim].contiguous())

    dst_model.model.layers[i].mlp.experts.group_w2.data = \
        torch.cat(down_proj_list, dim=0).view(expert_num, -1, ffn_dim).contiguous()
    group_gate_proj = torch.cat(gate_proj_list, dim=0).view(expert_num, ffn_dim, -1).contiguous()
    group_up_proj = torch.cat(up_proj_list, dim=0).view(expert_num, ffn_dim, -1).contiguous()
    dst_model.model.layers[i].mlp.experts.group_w1_w3.data = torch.cat([group_gate_proj, group_up_proj], dim=1)


def split_w_attn(block, dst_model, i, local_rank):
    qkv_dim = dst_model.model.layers[0].self_attn.num_heads_per_rank * \
                dst_model.model.layers[0].self_attn.head_dim + \
                (dst_model.model.layers[0].self_attn.num_key_value_heads_per_rank * \
                dst_model.model.layers[0].self_attn.head_dim) * 2
    o_dim = dst_model.model.layers[0].self_attn.attn_intermediate_size_per_rank

    q_proj_dim = dst_model.model.layers[0].self_attn.num_heads_per_rank * \
                dst_model.model.layers[0].self_attn.head_dim
    kv_proj_dim = dst_model.model.layers[0].self_attn.head_dim
    qkv_weight_list = []
    kv_rank = local_rank // dst_model.model.layers[0].self_attn.num_heads_per_rank
    q_proj_weight = block.self_attn.q_proj.weight.data[local_rank * q_proj_dim: (local_rank + 1) * q_proj_dim, :]
    k_proj_weight = block.self_attn.k_proj.weight.data[kv_rank * kv_proj_dim: (kv_rank + 1) * kv_proj_dim, :]
    v_proj_weight = block.self_attn.v_proj.weight.data[kv_rank * kv_proj_dim: (kv_rank + 1) * kv_proj_dim, :]
    qkv_weight_list.append(_to_parameter(torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], axis=0)))
    dst_model.model.layers[i].self_attn.merged_qkv_proj.weight = _to_parameter(torch.cat(qkv_weight_list, axis=0))

    dst_model.model.layers[i].self_attn.q_norm.weight.data = \
        block.self_attn.q_norm.weight.data
    dst_model.model.layers[i].self_attn.k_norm.weight.data = \
        block.self_attn.k_norm.weight.data
    dst_model.model.layers[i].self_attn.o_proj.weight.data = \
        block.self_attn.o_proj.weight.data[:, local_rank * o_dim: (local_rank + 1) * o_dim].contiguous()
    dst_model.model.layers[i].input_layernorm.weight.data = \
        block.input_layernorm.weight.data
    dst_model.model.layers[i].post_attention_layernorm.weight.data = \
        block.post_attention_layernorm.weight.data


def split_w(src_model, dst_model, world_size, local_rank):
    vocab_size = src_model.model.vocab_size // world_size

    dst_model.lm_head.weight.data = \
        src_model.lm_head.weight.data[local_rank * vocab_size: (local_rank + 1) * vocab_size, :]
    dst_model.model.embed_tokens.weight.data = \
        src_model.model.embed_tokens.weight.data[local_rank * vocab_size: (local_rank + 1) * vocab_size, :]

    dst_model.model.norm.weight.data = src_model.model.norm.weight.data

    for i, block in enumerate(src_model.model.layers):
        # attn weights
        split_w_attn(block, dst_model, i, local_rank)

        # moe experts weights
        split_w_moe(block, dst_model, i, local_rank)


def copy_files_with_prefix(src_dir, dst_dir, prefix):
    for file in os.listdir(src_dir):
        if file.startswith(prefix):
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Split weight parameters with tensor parallel")
    parser.add_argument('--model-path', type=str, help="Path of model weights")
    parser.add_argument('--output-path', type=str, help="The output directory where the results are saved")
    parser.add_argument('--world-size', type=int, default=8, help="The split times of model weights")
    parser_args = parser.parse_args()
    return parser_args


def show_model_states(origin_model, model_name="src_model"):
    src_param_size = 0
    for name, params in origin_model.named_parameters():
        size_per_param = np.prod(params.size())
        src_param_size += size_per_param
        logging.info("Param of %s tensor parallel: %s, %s, %s",
                     model_name, name, params.size(), params.dtype)
    logging.info("Total param size of %s tensor parallel: %s", model_name, src_param_size)


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output_path
    rank_size = args.world_size
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    origin_model = Qwen3MoeForCausalLMOri.from_pretrained(args.model_path,
                                                        ignore_mismatched_sizes=True,
                                                        low_cpu_mem_usage=True,
                                                        torch_dtype=torch.bfloat16)
    show_model_states(origin_model, "origin_model")

    for rank_id in range(rank_size):
        logging.info("rank_id={} / rank_size={}".format(rank_id, rank_size))
        os.environ["LOCAL_RANK"] = str(rank_id)

        save_path = os.path.join(output_path, f"rank_{rank_id}")
        logging.info("Split weight for rank %s start, save path is: %s", rank_id, save_path)

        config = origin_model.config
        part_model = Qwen3MoeForCausalLM(config)

        split_w(origin_model, part_model, rank_size, rank_id)

        show_model_states(part_model, "dst_model")

        part_model.save_pretrained(save_path)
        copy_files_with_prefix(args.model_path, save_path, "tokenizer")
        logging.info("Split weight for rank %s finished, save path is: %s", rank_id, save_path)

        del part_model