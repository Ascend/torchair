# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
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
import time
import argparse
import logging
import copy
import numpy as np
import torch
import torch_npu

from transformers import AutoTokenizer

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)

torch.manual_seed(42)
torch.npu.manual_seed_all(42)


class ModelRunner:
    def __init__(self, model_path, execute_mode, **kwargs):
        self.model_name = kwargs.get("model_name", "default_model_name")
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 131072)
        self.input_max_len = kwargs.get("input_max_len", 1024)
        self.max_new_tokens = kwargs.get("max_new_tokens", 32)
        self.batch_size = kwargs.get("batch_size", 72)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        if self.world_size == 1:
            self.model_path = model_path
        else:
            self.model_path = os.path.join(model_path, f"rank_{self.local_rank}")
        self.use_pretrained_model = True
        self.execute_mode = execute_mode
        self.tokenizer_mode = kwargs.get("tokenizer_mode", "default")
        self.init_device()
        self.enable_aclgraph = int(os.getenv("ENABLE_ACLGRAPH", "0"))

    @staticmethod
    def repeat_batch(tensor, repeat_num):
        if repeat_num == 1:
            return tensor
        return tensor.repeat(repeat_num, *[1] * (tensor.dim() - 1))

    def init_device(self):
        logging.info("Set execution using npu index: %s, global: %s", self.local_rank, self.global_rank)
        self.device = torch.device("%s:%s" % ("npu", self.local_rank))
        torch.npu.set_device(self.device)

        master_addr = os.environ["MASTER_ADDR"]
        master_port = int(os.environ["MASTER_PORT"])

        if torch.npu.is_available() and self.world_size > 1:
            torch.distributed.init_process_group(
                backend="hccl", world_size=self.world_size, rank=self.global_rank)

    def init_model(self, model, config=None):
        if self.use_pretrained_model:
            self.load_model(model)
        else:
            self.init_model_from_config(model, config=config)
        self.to_device()
        self.cast_format()
        self.compile_model()
        self.init_tokenizer()

    def init_model_from_config(self, model, config):
        config_file = "config.json"
        model_config = config.from_pretrained(config_file, torch_dtype=self.dtype,
                                              max_position_embeddings=self.max_position_embeddings)
        self.model = model(model_config).to(self.dtype)
    
    def load_model(self, model):
        logging.info("Try to load pretrained model in path: %s", self.model_path)
        self.model = model.from_pretrained(self.model_path,
                                            low_cpu_mem_usage=True,
                                            ignore_mismatched_sizes=True,
                                            torch_dtype=self.dtype,
                                            max_position_embeddings=self.max_position_embeddings)
    
    def save_model(self):
        pass

    def to_device(self):
        self.model.to(self.device)

    def cast_format(self):
        pass

    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="right", truncation_side='right')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def compile_model(self):
        logging.info("The final model structure is: \n %s", self.model)
        if self.execute_mode == "dynamo":
            logging.info("Try to compile model")
            self.graph_compile()

    def graph_compile(self):
        import torchair as tng
        import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
        from torchair.configs.compiler_config import CompilerConfig

        compiler_config = CompilerConfig()
        compiler_config.experimental_config.frozen_parameter = True
        compiler_config.experimental_config.tiling_schedule_optimize = True
        if self.enable_aclgraph:
            compiler_config.mode = "reduce-overhead"
        npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
        self.model = torch.compile(self.model, dynamic=True, fullgraph=True, backend=npu_backend)

    def mark_inputs(self, model_inputs):
        if self.execute_mode == "dynamo":
            pass

    def model_input_prepare(self, input_dict):
        pass

    def model_inference(self, model_inputs, warm_up=False):
        torch.npu.synchronize()
        if warm_up:
            self.mark_inputs(model_inputs)
        start_time = time.time()
        with torch.no_grad():
            logits = self.model(**model_inputs)
        torch.npu.synchronize()
        end_time = time.time()
        logging.info(f"{self.model_name} inference time cost {(end_time - start_time)*1000:.2f} ms")
        return logits
    
    def model_generate(self, prompts, warm_up=False, **kwargs):
        pass
    
    
    def model_output_process(self, model_inputs, outputs, input_dict):
        pass


