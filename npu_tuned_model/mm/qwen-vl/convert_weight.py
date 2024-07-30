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


def load_model(model_dir):
    stat_dict = dict()
    for file_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, file_name)
        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file as safe_load_file
            stat_dict.update(safe_load_file(model_path, device="cpu"))
        elif model_path.endswith((".bin", ".pth", ".pt")):
            stat_dict.update(torch.load(model_path, map_location="cpu"))
    return stat_dict


def save_model(stat_dict, model_dir, safe_serialization=False):
    if safe_serialization:
        from safetensors.torch import save_file
        save_file(stat_dict, f"{model_dir}/model.safetensors", metadata={"format": "pt"})
    else:
        torch.save(stat_dict, f"{model_dir}/pytorch_model.bin")


def replace_key(stat_dict, stat_dict_new):
    for key in stat_dict.keys():
        if "transformer.visual" in key:
            new_key = key.replace("transformer.visual", "visual")
        else:
            new_key = key
        stat_dict_new[new_key] = stat_dict[key]


def parse_args():
    parser = argparse.ArgumentParser(description="convert weight parameters")
    parser.add_argument('--model_path', type=str, help="Location of model weights")
    parser.add_argument('--output_path', type=str, help="The output directory where the results are saved")
    parser_args = parser.parse_args()
    return parser_args


def copy_files(src_dir, dst_dir, files_to_copy):
    for file in os.listdir(src_dir):
        if file in files_to_copy:
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    files = ["config.json", "generation_config.json", "qwen.tiktoken", "tokenizer_config.json"]
    copy_files(args.model_path, args.output_path, files)
    old_stat_dict = load_model(args.model_path)
    new_stat_dict = dict()
    replace_key(old_stat_dict, new_stat_dict)
    save_model(new_stat_dict, args.output_path)
