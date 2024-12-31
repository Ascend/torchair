import os
import argparse
import logging
import shutil
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from models.modeling_deepseek import DeepseekV3ForCausalLM

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)


def _to_parameter(data):
    return nn.Parameter(data, requires_grad=False)


def split_w_dense_dp(block, dst_model, i, locak_rank):
    up_weight_list = []

    gate_weight = block.mlp.gate_proj.weight
    up_weight = block.mlp.up_proj.weight
    up_weight_list.append(_to_parameter(torch.cat([gate_weight, up_weight], axis=0)))

    if len(up_weight_list) == 1:
        dst_model.model.layers[i].mlp.merge_up_gate_proj.weight = up_weight_list[0]
    else:
        dst_model.model.layers[i].mlp.merge_up_gate_proj.weight = _to_parameter(torch.cat(up_weight_list, axis=0))
    dst_model.model.layers[i].mlp.down_proj.weight.data = block.mlp.down_proj.weight.data.contiguous()


def split_w_dense(block, dst_model, i, local_rank):
    up_weight_list = []
    ffn_dim = dst_model.model.layers[i].mlp.intermediate_size_per_rank
    gate_weight = block.mlp.gate_proj.weight[local_rank * ffn_dim: (local_rank + 1) * ffn_dim, :].contiguous()
    up_weight = block.mlp.up_proj.weight[local_rank * ffn_dim: (local_rank + 1) * ffn_dim, :].contiguous()
    up_weight_list.append(_to_parameter(torch.cat([gate_weight, up_weight], axis=0)))

    if len(up_weight_list) == 1:
        dst_model.model.layers[i].mlp.merge_up_gate_proj.weight = up_weight_list[0]
    else:
        dst_model.model.layers[i].mlp.merge_up_gate_proj.weight = _to_parameter(torch.cat(up_weight_list, axis=0))
    dst_model.model.layers[i].mlp.down_proj.weight.data = \
        block.mlp.down_proj.weight.data[:, local_rank * ffn_dim: (local_rank + 1) * ffn_dim].contiguous()


def split_w_moe(block, dst_model, i, local_rank):
    shared_up_weight_list = []
    ffn_dim = dst_model.model.layers[i].mlp.shared_experts.intermediate_size_per_rank
    gate_weight = \
        block.mlp.shared_experts.gate_proj.weight[local_rank * ffn_dim: (local_rank + 1) * ffn_dim, :].contiguous()
    up_weight = \
        block.mlp.shared_experts.up_proj.weight[local_rank * ffn_dim: (local_rank + 1) * ffn_dim, :].contiguous()
    shared_up_weight_list.append(_to_parameter(torch.cat([gate_weight, up_weight], axis=0)))
    if len(shared_up_weight_list) == 1:
        dst_model.model.layers[i].mlp.shared_experts.merge_up_gate_proj.weight = shared_up_weight_list[0]
    else:
        dst_model.model.layers[i].mlp.shared_experts.merge_up_gate_proj.weight = \
            _to_parameter(torch.cat(shared_up_weight_list, axis=0))
    dst_model.model.layers[i].mlp.shared_experts.down_proj.weight.data = \
        block.mlp.shared_experts.down_proj.weight.data[:, local_rank * ffn_dim: (local_rank + 1) * ffn_dim].contiguous()
    dst_model.model.layers[i].mlp.gate.weight.data = block.mlp.gate.weight.data
    if dst_model.model.layers[i].mlp.gate.topk_method == "noaux_tc":
        dst_model.model.layers[i].mlp.gate.e_score_correction_bias.data = block.mlp.gate.e_score_correction_bias.data

    expert_num = block.mlp.config.n_routed_experts
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
    q_dim = dst_model.model.layers[0].self_attn.num_heads_per_rank * dst_model.model.layers[0].self_attn.q_head_dim
    o_dim = dst_model.model.layers[0].self_attn.num_heads_per_rank * dst_model.model.layers[0].self_attn.v_head_dim

    if dst_model.model.layers[i].self_attn.q_lora_rank is None:
        dst_model.model.layers[i].self_attn.q_proj.weight.data = \
            block.self_attn.q_proj.weight.data[local_rank * q_dim: (local_rank + 1) * q_dim, :].contiguous()
    else:
        dst_model.model.layers[i].self_attn.q_a_proj.weight.data = \
            block.self_attn.q_a_proj.weight.data
        dst_model.model.layers[i].self_attn.q_a_layernorm.weight.data = \
            block.self_attn.q_a_layernorm.weight.data
        dst_model.model.layers[i].self_attn.q_b_proj.weight.data = \
            block.self_attn.q_b_proj.weight.data[local_rank * q_dim: (local_rank + 1) * q_dim, :].contiguous()

    dst_model.model.layers[i].self_attn.kv_a_proj_with_mqa.weight.data = \
        block.self_attn.kv_a_proj_with_mqa.weight.data

    dst_model.model.layers[i].self_attn.kv_a_layernorm.weight.data = \
        block.self_attn.kv_a_layernorm.weight.data
    dst_model.model.layers[i].self_attn.o_proj.weight.data = \
        block.self_attn.o_proj.weight.data[:, local_rank * o_dim: (local_rank + 1) * o_dim].contiguous()
    dst_model.model.layers[i].input_layernorm.weight.data = \
        block.input_layernorm.weight.data
    dst_model.model.layers[i].post_attention_layernorm.weight.data = \
        block.post_attention_layernorm.weight.data


def kv_low_rank_split(block, dst_model, i, local_rank):
    k_dim = dst_model.model.layers[0].self_attn.num_heads_per_rank * \
                (dst_model.model.layers[0].self_attn.qk_nope_head_dim + dst_model.model.layers[0].self_attn.v_head_dim)
    kv_b_proj_weight_data = \
        block.self_attn.kv_b_proj.weight.data[local_rank * k_dim: (local_rank + 1) * k_dim, :].contiguous()
    qk_nope_head_dim = dst_model.model.layers[i].self_attn.qk_nope_head_dim
    num_heads_per_rank = dst_model.model.layers[i].self_attn.num_heads_per_rank
    kv_lora_rank = dst_model.model.layers[i].self_attn.kv_lora_rank
    v_head_dim = dst_model.model.layers[i].self_attn.v_head_dim

    index_tensor = torch.arange(qk_nope_head_dim).repeat(num_heads_per_rank) + \
        torch.arange(num_heads_per_rank).repeat_interleave(qk_nope_head_dim) * (qk_nope_head_dim + v_head_dim)
    kv_b_proj_w_k = torch.index_select(kv_b_proj_weight_data, dim=0, index=index_tensor)
    dst_model.model.layers[i].self_attn.kv_b_proj_w_k.data = \
        kv_b_proj_w_k.view(num_heads_per_rank, qk_nope_head_dim, kv_lora_rank).contiguous()
    index_tensor = torch.arange(qk_nope_head_dim, qk_nope_head_dim + v_head_dim).repeat(num_heads_per_rank) + \
        torch.arange(num_heads_per_rank).repeat_interleave(v_head_dim) * (qk_nope_head_dim + v_head_dim)
    kv_b_proj_w_v = torch.index_select(kv_b_proj_weight_data, dim=0, index=index_tensor)
    dst_model.model.layers[i].self_attn.kv_b_proj_w_v.data = \
        kv_b_proj_w_v.view(num_heads_per_rank, v_head_dim, kv_lora_rank).transpose(1, 2).contiguous()


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
        kv_low_rank_split(block, dst_model, i, local_rank)

        # moe experts weights
        if i >= dst_model.config.first_k_dense_replace and i % dst_model.config.moe_layer_freq == 0:
            split_w_moe(block, dst_model, i, local_rank)
        else:
            split_w_dense(block, dst_model, i, local_rank)


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
    origin_model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                        trust_remote_code=True,
                                                        ignore_mismatched_sizes=True,
                                                        low_cpu_mem_usage=True,
                                                        torch_dtype=torch.bfloat16,
                                                        attn_implementation="eager")
    show_model_states(origin_model, "origin_model")

    for rank_id in range(rank_size):
        logging.info("rank_id={} / rank_size={}".format(rank_id, rank_size))
        os.environ["LOCAL_RANK"] = str(rank_id)

        save_path = os.path.join(output_path, f"rank_{rank_id}")
        logging.info("Split weight for rank %s start, save path is: %s", rank_id, save_path)

        config = origin_model.config
        part_model = DeepseekV3ForCausalLM(config)

        split_w(origin_model, part_model, rank_size, rank_id)

        show_model_states(part_model, "dst_model")

        part_model.save_pretrained(save_path)
        copy_files_with_prefix(args.model_path, save_path, "tokenizer")
        logging.info("Split weight for rank %s finished, save path is: %s", rank_id, save_path)

        del part_model