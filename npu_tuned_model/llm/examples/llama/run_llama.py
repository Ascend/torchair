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
import argparse
import logging
import torch
from runner.llm_runner import LlmModelRunner

prompts = [
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
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank id")
    parser_args = parser.parse_args()
    return parser_args


def run_llama(model_path, **kwargs):
    # the first paramter should be changed to llama3 if running llama3 model
    model_runner = LlmModelRunner("llama2", model_path, **kwargs)
    model_runner.execute_mode = "dynamo"
    os.environ["EXE_MODE"] = model_runner.execute_mode
    model_runner.set_npu_config(jit_compile=False)
    model_runner.execute_model(prompts, **kwargs)


if __name__ == "__main__":
    args = parse_args()
    input_padding = True
    config = {
        "input_padding": input_padding,
        "dtype": torch.float16,
        "input_max_len": 1024,
        "max_new_tokens": 1024,
    }
    run_llama(args.model_path, **config)
    logging.info("model run success")
