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
import sys
import time
import argparse
import logging
import json
import torch

CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.realpath(os.path.join(CUR_DIR, ".."))
sys.path.append(ROOT_DIR)
from runner_deepseek import DeepSeekRunner

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)
torch.manual_seed(42)
torch.npu.manual_seed_all(42)


# basic token generater
def generate_default_prompt():
    # prompts的size大小决定了模型执行时的batch size大小
    preset_prompts = [
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
    return preset_prompts[0:1]


def generate_chat_prompt(bs):
    preset_prompts = [
        {"role": "user", "content": "Write a piece of quicksort code in C++"},
    ]
    preset_prompts = [preset_prompts] * (bs // len(preset_prompts) + 1)
    preset_prompts = preset_prompts[:bs]
    logging.info(f"chat prompt batch size: {bs}")
    return preset_prompts


def generate_prompt(bs, tokenizer_mode):
    if tokenizer_mode == "default":
        return generate_default_prompt()
    else:
        return generate_chat_prompt(bs)


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--model_path', type=str, help="Path of model weights")
    parser.add_argument('--model_name', type=str, help="Model name")
    parser.add_argument('--execute_mode', type=str, default="eager", choices=["dynamo", "eager"],
                        help="eager or dynamo")
    parser.add_argument('--tokenizer_mode', type=str, default="default", choices=["default", "chat"],
                        help="tokenizer_mode should be default or chat")
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank id for torch distributed launch")
    parser.add_argument('--input_max_len', type=int, default=1024, help="Max number of input")
    parser.add_argument('--max_new_tokens', type=int, default=32, help="Max number of new tokens")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for testing")
    parser.add_argument('--json_path', type=str, help="Path of settings")
    parser_args = parser.parse_args()
    return parser_args


def run_deepseek(model_path, execute_mode, **kwargs):
    preset_prompts = generate_prompt(1, args.tokenizer_mode)
    model_runner = DeepSeekRunner(model_path, execute_mode, **kwargs)
    # 表示在图模式下开启算子二进制复用，提高图模式下编译阶段性能
    torch.npu.set_compile_mode(jit_compile=False)
    model_runner.init_model()
    # warmup
    model_runner.model_generate(preset_prompts, warm_up=True, **kwargs)
    # generate perf data
    model_runner.model_generate(preset_prompts, **kwargs)


if __name__ == "__main__":
    args = parse_args()
    input_max_len = args.input_max_len # 输入padding的长孺
    max_new_tokens = args.max_new_tokens # 最大输出token的个数
    max_position_embeddings = input_max_len + max_new_tokens # 用于申请kv_cache时指定seq_len长度
    model_config = {
        "dtype": torch.bfloat16,
        "input_max_len": input_max_len,
        "max_new_tokens": max_new_tokens,
        "max_position_embeddings": max_position_embeddings
    }
    run_config = {
        "tokenizer_mode": args.tokenizer_mode,
        "batch_size": args.batch_size,
        "model_name": args.model_name
    }
    config = {**model_config, **run_config}
    os.environ["EXE_MODE"] = args.execute_mode
    run_deepseek(args.model_path, args.execute_mode, **config)
    logging.info("model run success")
