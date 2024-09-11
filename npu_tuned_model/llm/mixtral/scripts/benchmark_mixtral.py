# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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
import time
import argparse
import logging
import torch

from transformers import AutoTokenizer
from models.modeling_mixtral import MixtralForCausalLM

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)


class ModelRunner:
    def __init__(self, model_path, execute_mode, **kwargs):
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 131072)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.model_path = os.path.join(model_path, f"rank_{self.local_rank}")
        self.execute_mode = execute_mode

    def init_model(self):
        logging.info("Set execution using npu index: %s", self.local_rank)
        self.device = torch.device("%s:%s" % ("npu", self.local_rank))
        torch.npu.set_device(self.device)

        if torch.npu.is_available() and self.world_size > 1:
            torch.distributed.init_process_group(
                backend="hccl", world_size=self.world_size, rank=self.local_rank)

        logging.info("Try to load pretrained model in path: %s", self.model_path)
        self.model = MixtralForCausalLM.from_pretrained(self.model_path,
                                                low_cpu_mem_usage=True,
                                                torch_dtype=self.dtype,
                                                max_position_embeddings=self.max_position_embeddings)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="right", truncation_side='right')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.to(self.device)
        logging.info("The final model structure is: \n %s", self.model)
        if self.execute_mode == "dynamo":
            logging.info("Try to compile model")
            self.graph_compile()
    
    def graph_compile(self):
        import torchair as tng
        import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
        from torchair.configs.compiler_config import CompilerConfig

        compiler_config = CompilerConfig()
        compiler_config.experimental_config.frozen_parameter = True
        compiler_config.experimental_config.tiling_schedule_optimize = True
        npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
        self.model.model = torch.compile(self.model.model, dynamic=True, fullgraph=True, backend=npu_backend)
    
    def mark_inputs(self, model_inputs):
        if self.execute_mode == "dynamo":
            input_ids = model_inputs.get("input_ids")
            kv_len = model_inputs.get("kv_len")
            attention_mask = model_inputs.get("attention_mask")
            position_ids = model_inputs.get("position_ids")
            past_key_values = model_inputs.get("past_key_values")

            # prefill with dynamic sequence length, decode with static sequence length
            torch._dynamo.mark_static(kv_len)
            for item in past_key_values:
                for sub_item in item:
                    torch._dynamo.mark_static(sub_item)
            
            if input_ids.shape[1] == 1:
                torch._dynamo.mark_static(input_ids)
                if attention_mask is not None:
                    torch._dynamo.mark_static(attention_mask)
                torch._dynamo.mark_static(position_ids)
            else:
                torch._dynamo.mark_static(kv_len)

    def model_generate(self, prompts, warm_up=False, **kwargs):
        inputs = self.tokenizer(prompts, return_tensors="pt", truncation=True, padding='max_length',
                                max_length=kwargs.get("input_max_len", 1024)).to(self.device)

        input_ids = inputs.input_ids
        batch_size, _ = input_ids.size()
        attention_mask = inputs.attention_mask
        is_prefill = True
        generate_tokens = 0
        past_key_values = None
        prefill_time = 0
        decode_time = 0
        kv_len = None
        generate_ids = input_ids
        share_mask_tril = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=input_ids.device))

        while True:
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask,
                                past_key_values=past_key_values, is_prefill=is_prefill, kv_len=kv_len,
                                share_mask_tril=share_mask_tril, world_size=self.world_size)
            
            torch.npu.synchronize()
            start_time = time.time()
            if warm_up:
                self.mark_inputs(model_inputs)
            with torch.no_grad():
                logits = self.model(**model_inputs)
            torch.npu.synchronize()
            end_time = time.time()
            kv_len = torch.max(model_inputs.get("position_ids"), axis=1)[0] + 1
            attention_mask = model_inputs.get("attention_mask")
            past_key_values = model_inputs.get("past_key_values")
            next_tokens = torch.argmax(logits, dim=-1)
            generate_tokens += 1
            is_prefill = False

            if generate_tokens == 1:
                prefill_time += end_time - start_time
            else:
                decode_time += end_time - start_time
            if generate_tokens >= max_new_tokens:
                break
            
            generate_ids = torch.cat([generate_ids, next_tokens], dim=-1)
            input_ids = next_tokens

        res = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        if self.local_rank == 0:
            logging.info("Prefill inference time cost: %.2fms", prefill_time * 1000)
            if generate_tokens > 1:
                logging.info("Avg decode inference time cost: %.2fms", decode_time / (generate_tokens - 1) * 1000)
            if isinstance(res, list):
                for answer in res:
                    logging.info("Inference decode result: \n%s", answer)
            else:
                logging.info("Inference decode result: \n%s", res)


# prompts的size大小决定了模型执行时的batch size大小
_PROMPTS = [
    "用一句话描述地球为什么是独一无二的。",
    "给出一段对话，使用合适的语气和回答方式继续对话。\n对话：\nA：你今天看起来很高兴，发生了什么好事？\nB：是的，我刚刚得到一份来自"
    "梅西银行的工作通知书。\nA：哇，恭喜你！你打算什么时候开始工作？\nB：下个月开始，所以我现在正为这份工作做准备。",
    "Let x = 1. What is x << 3 in Python 3? the answer is",
    "In Python 3, what is ['a', 'Chemistry', 0, 1][-3]?",
    "The study of older adults and aging is reffered to as",
    "Why is the sky blue?",
    "What's your name?",
    "Hello my name is",
]


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--model-path', type=str, help="Path of model weights")
    parser.add_argument('--execute-mode', type=str, default="dynamo", choices=["dynamo", "eager"],
                        help="eager or dynamo")
    parser.add_argument('--local-rank', type=int, default=0, help="Local rank id")
    parser_args = parser.parse_args()
    return parser_args


def run_mixtral(model_path, execute_mode, **kwargs):
    model_runner = ModelRunner(model_path, execute_mode, **kwargs)
    # 表示在图模式下开启算子二进制复用，提高图模式下编译阶段性能
    torch.npu.set_compile_mode(jit_compile=False)
    model_runner.init_model()
    # warmup
    model_runner.model_generate(_PROMPTS, warm_up=True, **kwargs)
    # generate perf data
    model_runner.model_generate(_PROMPTS, **kwargs)


if __name__ == "__main__":
    args = parse_args()
    input_max_len = 1024  # 输入padding的长度
    max_new_tokens = 32  # 最大输出token个数
    max_position_embeddings = (input_max_len + max_new_tokens)  # 用于申请kv_cache时指定seq_len长度
    config = {
        "dtype": torch.bfloat16,  # 和模型权重目录下config.json中torch_dtype一致
        "input_max_len": input_max_len,
        "max_new_tokens": max_new_tokens,
        "max_position_embeddings": max_position_embeddings
    }
    os.environ["EXE_MODE"] = args.execute_mode
    run_mixtral(args.model_path, args.execute_mode, **config)
    logging.info("model run success")
