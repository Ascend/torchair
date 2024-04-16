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
    "Common sense questions\n\nQuestion:What is a banana?",
    "Common sense questions and answers\n\nQuestion: What is a dog?\nFactual answer:",
    "Common sense questions and answers\nQuestion: How to learn a language?\nFactual answer:",
    "Common sense questions and answers\n\nQuestion: Can you introduce yourself?\nFactual answer:",
]


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--model_path', type=str, help="Location of model weights")
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank id")
    parser_args = parser.parse_args()
    return parser_args


def run_llama2(model_path, **kwargs):
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
    run_llama2(args.model_path, **config)
    logging.info("model run success")
