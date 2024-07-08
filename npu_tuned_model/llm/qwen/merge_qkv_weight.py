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
import shutil
import argparse
import torch
from modeling_qwen2 import Qwen2ForCausalLM, Qwen2DecoderLayer


def parse_args():
    parser = argparse.ArgumentParser(description="convert weight parameters")
    parser.add_argument('--model_path', type=str, help="Location of model weights")
    parser.add_argument('--model_name', type=str, default="llama2", help="model name, e.g., llama2")
    parser.add_argument('--tp_size', type=int, default=1, help="The number of tensor split")
    parser.add_argument('--output_path', type=str, help="The output directory where the results are saved")
    parser_args = parser.parse_args()
    return parser_args


def copy_files_with_prefix(src_dir, dst_dir, prefixes):
    for file in os.listdir(src_dir):
        for prefix in prefixes:
            if file.startswith(prefix):
                src_file = os.path.join(src_dir, file)
                dst_file = os.path.join(dst_dir, file)
                shutil.copy2(src_file, dst_file)


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    file_prefixes = ["tokenizer", "merges", "vocab"]
    copy_files_with_prefix(args.model_path, args.output_path, file_prefixes)
    model = Qwen2ForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    model.merge_qkv_weight(args.tp_size)
    model.save_pretrained(args.output_path, from_pt=True)
