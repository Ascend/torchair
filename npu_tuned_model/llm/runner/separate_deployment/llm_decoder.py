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

import torch
from runner.separate_deployment.llm_inference import SeparateDeployModelRunner


class LlmDecoderRunner(SeparateDeployModelRunner):
    def __init__(self, model_name, model_path, **kwargs):
        super().__init__(model_name=model_name, model_path=model_path, **kwargs)

    def prepare_decoder_inputs(self, outputs_param, **kwargs):
        if (outputs_param.get("input_ids", None) is None or outputs_param.get("generate_ids", None) is None or
                outputs_param.get("attention_mask", None) is None):
            raise ValueError("input_ids or generate_ids or attention_mask is None")
        kwargs_params = self._generate_params(outputs_param)
        kwargs_params["input_ids"] = torch.cat([kwargs_params["input_ids"], kwargs_params["generate_ids"]], dim=-1)
        attention_mask = kwargs_params["attention_mask"]
        kwargs_params["attention_mask"] = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

        return kwargs_params

    @torch.inference_mode()
    def execute_model(self, inputs, **kwargs):
        model_inputs = self.model.prepare_inputs_for_generation(input_ids=inputs["input_ids"],
                                                                attention_mask=inputs["attention_mask"],
                                                                **kwargs)
        outputs = self.model_generate(model_inputs)
        return outputs
