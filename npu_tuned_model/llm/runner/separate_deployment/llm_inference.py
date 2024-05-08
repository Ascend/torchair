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
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from runner.common_runner import LlmCommonModelRunner


class SeparateDeployModelRunner(LlmCommonModelRunner):
    def __init__(self, model_name, model_path, **kwargs):
        self.cnt = 0
        self.run_time = 0
        super().__init__(model_name=model_name, model_path=model_path, **kwargs)

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
        dynamic_compile = eval(os.getenv("DYNAMIC_COMPILE", "False"))
        if self._execute_mode == "dynamo":
            logging.info(f"Start to run model in dynamo mode, dynamic={dynamic_compile}, fullgraph=True, backend=npu")
            config = CompilerConfig()
            config.experimental_config.frozen_parameter = True
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
