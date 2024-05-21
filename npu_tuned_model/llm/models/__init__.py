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

import importlib
from typing import List, Optional, Type

import torch.nn as nn

# model name -> (module, class).
_MODELS = {
    "llama2": ("modeling_llama", "LlamaForCausalLM"),
    "llama3": ("modeling_llama", "LlamaForCausalLM"),
}

# model injection_policy
_MODEL_INJECTION_POLICY = {
    "llama2": ("modeling_llama", "LlamaDecoderLayer", ('self_attn.o_proj', 'mlp.down_proj')),
    "llama3": ("modeling_llama", "LlamaDecoderLayer", ('self_attn.o_proj', 'mlp.down_proj')),
}


class ModelRegistry:
    @staticmethod
    def load_model_cls(model_name: str) -> Optional[Type[nn.Module]]:
        if model_name not in _MODELS:
            return None

        module_file, model_cls_name = _MODELS[model_name]
        model_module_name = model_name
        if model_name == "llama2" or model_name == "llama3":
            model_module_name = "llama2_llama3"
        module = importlib.import_module(
            f"models.{model_module_name}.{module_file}")
        return getattr(module, model_cls_name, None)

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys())

    @staticmethod
    def get_model_injection_policy(model_name: str):
        if model_name not in _MODEL_INJECTION_POLICY:
            return None

        module_file, model_cls_name, layers = _MODEL_INJECTION_POLICY[model_name]
        if model_name == "llama2" or model_name == "llama3":
            model_module_name = "llama2_llama3"
        module = importlib.import_module(
            f"models.{model_module_name}.{module_file}")
        model_cls = getattr(module, model_cls_name, None)
        return {model_cls: layers}


__all__ = [
    "ModelRegistry",
]
