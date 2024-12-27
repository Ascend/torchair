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
import deepspeed
from transformers import AutoTokenizer
from modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig


class SeparateDeployModelRunner:
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.dtype = kwargs.get("dtype", torch.float16)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_logging = (self.local_rank == 0)
        self.cnt = 0
        self.run_time = 0

    def init_model(self):
        logging.info("Set execution using npu index: %s", self.local_rank)
        self.device = torch.device("%s:%s" % ("npu", self.local_rank))
        torch.npu.set_device(self.device)

        logging.info("Try to load pretrained model in path: %s", self.model_path) if self.is_logging else None
        self.model = LlamaForCausalLM.from_pretrained(self.model_path, low_cpu_mem_usage=True, torch_dtype=self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        deepspeed.init_distributed(dist_backend="hccl")
        if os.getenv("EXE_MODE", None) == "dynamo":
            import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce

        deepspeed_model = deepspeed.init_inference(
            model=self.model,
            mp_size=self.world_size,
            dtype=self.dtype,
            relace_method="auto",
            replace_with_kernel_inject=False,
            injection_policy={LlamaDecoderLayer: ('self_attn.o_proj', 'mlp.down_proj')}
        )
        deepspeed_model.to(self.device)
        self.model.to(self.device)
        logging.info("The final model structure is: \n %s", self.model) if self.is_logging else None

    def process_output(self, generate_ids):
        res = self.tokenizer.batch_decode(generate_ids,
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
        return res

    def model_generate(self, model_inputs):
        torch.npu.synchronize()
        start = time.time()
        outputs = self.model(**model_inputs)
        torch.npu.synchronize()
        end = time.time()
        self.cnt += 1
        self.run_time += (end - start)
        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        return next_tokens

    def inference_avg_time(self):
        if self.cnt == 0:
            return 0
        return self.run_time / self.cnt

    def reset_time_statistics(self):
        self.cnt = 0
        self.run_time = 0

    @torch.inference_mode()
    def execute_model(self, prompts, **kwargs):
        raise NotImplementedError

    def compile_model(self):
        if os.getenv("EXE_MODE") == "dynamo":
            dynamic_compile = True # 因为当模型结构使能了actual_seq_length
            logging.info(f"Start to run model in dynamo mode, dynamic={dynamic_compile}, fullgraph=True, backend=npu")
            config = CompilerConfig()
            config.experimental_config.frozen_parameter = True
            config.experimental_config.tiling_schedule_optimize = True # tiling全下沉性能优化
            npu_backend = tng.get_npu_backend(compiler_config=config)
            self.model = torch.compile(self.model, dynamic=dynamic_compile, fullgraph=True, backend=npu_backend)
        else:
            logging.info("Start to run model in eager(HOST API) mode")

    def _generate_params(self, inputs):
        kwargs_params = {}
        for key in inputs.keys():
            value = inputs[key]
            if isinstance(value, torch.Tensor):
                kwargs_params.update({
                    key: value.to(self.device)
                })
            else:
                kwargs_params.update({
                    key: value
                })

        return kwargs_params
