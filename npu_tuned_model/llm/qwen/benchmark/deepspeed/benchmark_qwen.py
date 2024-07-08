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
import deepspeed

from transformers import AutoTokenizer
from modeling_qwen2 import Qwen2ForCausalLM, Qwen2DecoderLayer

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)


class ModelRunner:
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 131072)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_logging = (self.local_rank == 0)

    def init_model(self):
        logging.info("Set execution using npu index: %s", self.local_rank)
        self.device = torch.device("%s:%s" % ("npu", self.local_rank))
        torch.npu.set_device(self.device)

        logging.info("Try to load pretrained model in path: %s", self.model_path) if self.is_logging else None
        self.model = Qwen2ForCausalLM.from_pretrained(self.model_path,
                                                      low_cpu_mem_usage=True,
                                                      torch_dtype=self.dtype,
                                                      max_position_embeddings=self.max_position_embeddings)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        deepspeed.init_distributed(dist_backend="hccl")
        if os.getenv("EXE_MODE", "dynamo") == "dynamo":
            import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce

        deepspeed_model = deepspeed.init_inference(
            model=self.model,
            mp_size=self.world_size,
            dtype=self.dtype
        )
        deepspeed_model.to(self.device)
        self.model.to(self.device)
        logging.info("The final model structure is: \n %s", self.model)

    def model_generate(self, prompts, decode_out=False, **kwargs):
        inputs = self.tokenizer(prompts,
                                return_tensors="pt",  # 返回pytorch tensor
                                truncation=True,
                                padding='max_length',
                                max_length=kwargs.get("input_max_len", 1024))

        kwargs_params = self._generate_params(inputs, kwargs.get("max_new_tokens", 1024))
        start_time = time.time()
        generate_ids = self.model.generate(**kwargs_params)
        elapse = time.time() - start_time
        logging.info("Model execute success, time cost: %.2fs", elapse) if self.is_logging else None

        if not decode_out:
            return
        input_tokens = len(inputs.input_ids[0])
        output_tokens = len(generate_ids[0])
        new_tokens = output_tokens - input_tokens
        res = self.tokenizer.batch_decode(generate_ids[:, input_tokens:],
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)

        if self.is_logging:
            if isinstance(res, list):
                for answer in res:
                    logging.info("Inference decode result: \n%s", answer)
            else:
                logging.info("Inference decode result: \n%s", res)
            logging.info("Output tokens number: %s, input tokens number:%s, total new tokens generated: %s",
                         output_tokens, input_tokens, new_tokens)

    def _generate_params(self, inputs, max_new_tokens):
        kwargs_params = {"max_new_tokens": max_new_tokens}
        for key in inputs.keys():
            kwargs_params.update({
                key: inputs[key].to(self.device)
            })
        return kwargs_params


# prompts的size大小决定了模型执行时的batch size大小
_PROMPTS = [
    "用一句话描述地球为什么是独一无二的。",
    "给出一段对话，使用合适的语气和回答方式继续对话。\n对话：\nA：你今天看起来很高兴，发生了什么好事？\nB：是的，我刚刚得到一份来自"
    "梅西银行的工作通知书。\nA：哇，恭喜你！你打算什么时候开始工作？\nB：下个月开始，所以我现在正为这份工作做准备。",
    "基于以下提示填写以下句子的空格。\n提示：\n- 提供多种现实世界的场景\n- 空格应填写一个形容词或一个形容词短语\n句子:\n______出去"
    "享受户外活动，包括在公园里散步，穿过树林或在海岸边散步。",
    "请生成一个新闻标题，描述一场正在发生的大型自然灾害。",
]


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--model_path', type=str, help="Location of model weights")
    parser.add_argument('--execute_mode', type=str, default="dynamo", choices=["dynamo", "eager"],
                        help="eager or dynamo")
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank id")
    parser_args = parser.parse_args()
    return parser_args


def run_qwen(model_path, **kwargs):
    model_runner = ModelRunner(model_path, **kwargs)
    # 表示在图模式下开启算子二进制复用，提高图模式下编译阶段性能
    torch.npu.set_compile_mode(jit_compile=False)
    model_runner.init_model()
    # warmup
    model_runner.model_generate(_PROMPTS, **kwargs)
    # generate perf data
    model_runner.model_generate(_PROMPTS, decode_out=True, **kwargs)


if __name__ == "__main__":
    args = parse_args()
    input_max_len = 1024  # 输入padding的长度
    max_new_tokens = 1024  # 最大输出token个数
    max_position_embeddings = (input_max_len + max_new_tokens)  # 用于申请kv_cache时指定seq_len长度
    config = {
        "dtype": torch.bfloat16,  # 和模型权重目录下config.json中torch_dtype一致
        "input_max_len": input_max_len,
        "max_new_tokens": max_new_tokens,
        "max_position_embeddings": max_position_embeddings
    }
    os.environ["EXE_MODE"] = args.execute_mode
    run_qwen(args.model_path, **config)
    logging.info("model run success")
