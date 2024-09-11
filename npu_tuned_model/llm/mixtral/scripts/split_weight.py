import os
import argparse
import logging
import shutil
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from models.modeling_mixtral import MixtralForCausalLM

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)


def split_w(src_model, dst_model, world_size, local_rank, use_qkv_fusion=True, use_gmm_kernel=True):
    dst_model.lm_head.weight.data = src_model.lm_head.weight.data
    dst_model.model.embed_tokens.weight.data = src_model.model.embed_tokens.weight.data
    dst_model.model.norm.weight.data = src_model.model.norm.weight.data
    q_dim = dst_model.model.layers[0].self_attn.num_heads_per_rank * \
                dst_model.model.layers[0].self_attn.head_dim
    k_dim = dst_model.model.layers[0].self_attn.num_key_value_heads_per_rank * \
                dst_model.model.layers[0].self_attn.head_dim
    
    for i, block in enumerate(src_model.model.layers):
        if use_qkv_fusion:
            dst_model.model.layers[i].self_attn.c_attn.weight.data = \
                torch.cat([block.self_attn.q_proj.weight.data[local_rank * q_dim: (local_rank + 1) * q_dim, :],
                        block.self_attn.k_proj.weight.data[local_rank * k_dim: (local_rank + 1) * k_dim, :],
                        block.self_attn.v_proj.weight.data[local_rank * k_dim: (local_rank + 1) * k_dim, :]],
                        axis=0).contiguous()
        else:
            dst_model.model.layers[i].self_attn.q_proj.weight.data = \
                block.self_attn.q_proj.weight.data[local_rank * q_dim: (local_rank + 1) * q_dim, :].contiguous()
            dst_model.model.layers[i].self_attn.k_proj.weight.data = \
                block.self_attn.k_proj.weight.data[local_rank * k_dim: (local_rank + 1) * k_dim, :].contiguous()
            dst_model.model.layers[i].self_attn.v_proj.weight.data = \
                block.self_attn.v_proj.weight.data[local_rank * k_dim: (local_rank + 1) * k_dim, :].contiguous()
        dst_model.model.layers[i].self_attn.o_proj.weight.data = \
            block.self_attn.o_proj.weight.data[:, local_rank * q_dim: (local_rank + 1) * q_dim].contiguous()
        
        dst_model.model.layers[i].block_sparse_moe.gate.weight.data = block.block_sparse_moe.gate.weight.data
        dst_model.model.layers[i].input_layernorm.weight.data = block.input_layernorm.weight.data
        dst_model.model.layers[i].post_attention_layernorm.weight.data = block.post_attention_layernorm.weight.data
        
        expert_num = block.block_sparse_moe.num_experts
        w1_list, w2_list, w3_list = [], [], []
        for j, expert in enumerate(block.block_sparse_moe.experts):
            if use_gmm_kernel:
                ffn_dim = dst_model.model.layers[i].block_sparse_moe.experts.ffn_dim
                w1_list.append(expert.w1.weight.data[local_rank * ffn_dim: (local_rank + 1) * ffn_dim, :].contiguous())
                w2_list.append(expert.w2.weight.data[:, local_rank * ffn_dim: (local_rank + 1) * ffn_dim].contiguous())
                w3_list.append(expert.w3.weight.data[local_rank * ffn_dim: (local_rank + 1) * ffn_dim, :].contiguous())
            else:
                ffn_dim = dst_model.model.layers[i].block_sparse_moe.experts[j].ffn_dim
                dst_model.model.layers[i].block_sparse_moe.experts[j].w1.weight.data = \
                    expert.w1.weight.data[local_rank * ffn_dim: (local_rank + 1) * ffn_dim, :].contiguous()
                dst_model.model.layers[i].block_sparse_moe.experts[j].w3.weight.data = \
                    expert.w3.weight.data[local_rank * ffn_dim: (local_rank + 1) * ffn_dim, :].contiguous()
                dst_model.model.layers[i].block_sparse_moe.experts[j].w2.weight.data = \
                    expert.w2.weight.data[:, local_rank * ffn_dim: (local_rank + 1) * ffn_dim].contiguous()

        if use_gmm_kernel:
            dst_model.model.layers[i].block_sparse_moe.experts.group_w2.data = \
                torch.cat(w2_list, dim=0).view(expert_num, -1, ffn_dim).contiguous()
            group_w1 = torch.cat(w1_list, dim=0).view(expert_num, ffn_dim, -1).contiguous()
            group_w3 = torch.cat(w3_list, dim=0).view(expert_num, ffn_dim, -1).contiguous()
            dst_model.model.layers[i].block_sparse_moe.experts.group_w1_w3.data = torch.cat([group_w1, group_w3], dim=1)


def copy_files_with_prefix(src_dir, dst_dir, prefix):
    for file in os.listdir(src_dir):
        if file.startswith(prefix):
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)


def parse_args():
    parser = argparse.ArgumentParser(description="split weight parameters with tensor parallel")
    parser.add_argument('--model-path', type=str, help="Path of model weights")
    parser.add_argument('--output-path', type=str, help="The output directory where the results are saved")
    parser_args = parser.parse_args()
    return parser_args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    rank_id = int(os.getenv("LOCAL_RANK", "0"))
    rank_size = int(os.getenv("WORLD_SIZE", "1"))
    save_path = os.path.join(args.output_path, f"rank_{rank_id}")
    logging.info("Split weight for rank %s start, save path is: %s", rank_id, save_path)
    
    origin_model = AutoModelForCausalLM.from_pretrained(args.model_path,
                    low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    config = origin_model.config
    part_model = MixtralForCausalLM(config)

    src_param_size = 0
    for name, params in origin_model.named_parameters():
        size_per_param = np.prod(params.size())
        src_param_size += size_per_param
        logging.info("Param before tensor parallel: %s, %s, %s",
                     name, params.size(), params.dtype)
    logging.info("Total param size before tensor parallel: %s", src_param_size)
    split_w(origin_model, part_model, rank_size, rank_id)
    
    dst_param_size = 0
    for name, params in part_model.named_parameters():
        size_per_param = np.prod(params.size())
        dst_param_size += size_per_param
        logging.info("Param after tensor parallel: %s, %s, %s",
                     name, params.size(), params.dtype)
    logging.info("Total param size after tensor parallel: %s", dst_param_size)

    part_model.save_pretrained(save_path)
    copy_files_with_prefix(args.model_path, save_path, "tokenizer")
    logging.info("Split weight for rank %s finished, save path is: %s", rank_id, save_path)
