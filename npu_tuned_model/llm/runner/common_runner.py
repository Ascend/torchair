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
import logging
import time
import torch
import torch_npu

from transformers import AutoTokenizer

from models import ModelRegistry


class LlmCommonModelRunner:
    def __init__(self, model_name, model_path, **kwargs):
        self.model_name = model_name
        self.model_path = model_path
        self.input_padding = kwargs.get("input_padding", False)
        self.dtype = kwargs.get("dtype", torch.float16)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_logging = (self.local_rank == 0)
        self._execute_mode = "dynamo"

    @property
    def execute_mode(self):
        return self._execute_mode

    @staticmethod
    def set_npu_config(**kwargs):
        torch.npu.set_compile_mode(jit_compile=kwargs.get("jit_compile", False))
        npu_options = {"NPU_FUZZY_COMPILE_BLACKLIST": "ReduceProd"}
        torch.npu.set_option(npu_options)

    @execute_mode.setter
    def execute_mode(self, value):
        if value not in ["dynamo", "eager"]:
            raise ValueError(f"Unsupported execute mode:{value}")
        self._execute_mode = value

    def generate_tokenizer(self):
        if self.input_padding:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, padding_side="left")
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)

        return tokenizer

    def init_model(self):
        logging.info("Set execution using npu index: %s", self.local_rank)
        self.device = torch.device("%s:%s" % ("npu", self.local_rank))

        torch.npu.set_device(self.device)

        if self.is_logging:
            logging.info("Try to load pretrained model in path: %s", self.model_path)

        model_cls = ModelRegistry.load_model_cls(self.model_name)
        self.model = model_cls.from_pretrained(self.model_path, low_cpu_mem_usage=True, torch_dtype=self.dtype)
        self.model.input_padding = self.input_padding

        self.tokenizer = self.generate_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.world_size > 1:
            import deepspeed
            deepspeed.init_distributed(dist_backend="hccl")
            if self._execute_mode == "dynamo":
                import torchair._ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce

            deepspeed_model = deepspeed.init_inference(
                model=self.model,
                mp_size=self.world_size,
                dtype=self.dtype,
                replace_with_kernel_inject=False,
                injection_policy=ModelRegistry.get_model_injection_policy(self.model_name)
            )
            deepspeed_model.to(self.device)
        self.model.to(self.device)
        logging.info("The final model structure is: \n %s", self.model)

        if self._execute_mode == "dynamo":
            torch._dynamo.reset()


    def execute_model(self, prompts, **kwargs):
        raise NotImplementedError
