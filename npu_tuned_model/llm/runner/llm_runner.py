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

import logging
import time
import torch

from runner.common_runner import LlmCommonModelRunner


class LlmModelRunner(LlmCommonModelRunner):
    def __init__(self, model_name, model_path, **kwargs):
        super().__init__(model_name=model_name, model_path=model_path, **kwargs)

    def model_generate(self, prompts, decode_out=False, **kwargs):
        if self.input_padding:
            inputs = self.tokenizer(prompts,
                                    return_tensors="pt",  # 返回pytorch tensor
                                    truncation=True,
                                    padding='max_length',
                                    max_length=kwargs.get("input_max_len", 1024))
        else:
            inputs = self.tokenizer(prompts,
                                    return_tensors="pt",  # 返回pytorch tensor
                                    truncation=True)

        kwargs_params = self._generate_params(inputs, kwargs.get("max_new_tokens", 1024))
        start_time = time.time()
        with torch.no_grad():
            generate_ids = self.model.generate(**kwargs_params)
        elapse = time.time() - start_time
        if self.is_logging:
            logging.info("Model execute success, time cost: %.2fs", elapse)

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

    def warmup(self, warmup_prompts, **kwargs):
        self.model_generate(warmup_prompts, False, **kwargs)

    def execute_model(self, prompts, **kwargs):
        self.init_model()

        self._generate_answer(prompts, **kwargs)

        return 0

    def _generate_answer(self, prompts, **kwargs):
        self.warmup(prompts, **kwargs)

        # execute inference
        self.model_generate(prompts, True, **kwargs)

    def _generate_params(self, inputs, max_new_tokens):
        kwargs_params = {"max_new_tokens": max_new_tokens}
        for key in inputs.keys():
            kwargs_params.update({
                key: inputs[key].to(self.device)
            })
        return kwargs_params
