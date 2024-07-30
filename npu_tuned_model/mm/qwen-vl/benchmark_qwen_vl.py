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
import time
import argparse
import logging
import torch

from model.tokenization_qwen import QWenTokenizer
from model.modeling_qwen import QWenLMHeadModel

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)


class ModelRunner:
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 8192)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.local_rank = 0

    def init_model(self):
        logging.info("Set execution using npu index: %s", self.local_rank)
        self.device = torch.device("%s:%s" % ("npu", self.local_rank))
        torch.npu.set_device(self.device)
        logging.info("Try to load pretrained model in path: %s", self.model_path)
        self.model = QWenLMHeadModel.from_pretrained(self.model_path,
                                                     low_cpu_mem_usage=True,
                                                     torch_dtype=self.dtype,
                                                     max_position_embeddings=self.max_position_embeddings)
        self.tokenizer = QWenTokenizer.from_pretrained(self.model_path,
                                                       use_fast=False,
                                                       padding_side="left",
                                                       pad_token='<|endoftext|>')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.to(self.device)
        logging.info("The final model structure is: \n %s", self.model)

    def model_generate(self, prompts, decode_out=False, **kwargs):
        queries = list()
        for prompt in prompts:
            query = self.tokenizer.from_list_format(prompt)
            queries.append(query)
        inputs = self.tokenizer(queries,
                                return_tensors="pt",  # 返回pytorch tensor
                                truncation=True,
                                padding='max_length',
                                max_length=kwargs.get("input_max_len", 1024))

        kwargs_params = self._generate_params(inputs, kwargs.get("max_new_tokens", 1024))
        start_time = time.time()
        generate_ids = self.model.generate(**kwargs_params)
        elapse = time.time() - start_time
        logging.info("Model execute success, time cost: %.2fs", elapse)

        if not decode_out:
            return
        input_tokens = len(inputs.input_ids[0])
        output_tokens = len(generate_ids[0])
        new_tokens = output_tokens - input_tokens
        res = self.tokenizer.batch_decode(generate_ids[:, input_tokens:],
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)

        if isinstance(res, list):
            for answer in res:
                logging.info("Inference decode result: \n%s", answer)
        else:
            logging.info("Inference decode result: \n%s", res)
        logging.info("Output tokens number: %s, input tokens number:%s, total new tokens generated: %s",
                     output_tokens, input_tokens, new_tokens)

    def _generate_params(self, inputs, max_new_tokens):
        kwargs_params = {"max_new_tokens": max_new_tokens}
        for key in inputs.keys():
            kwargs_params.update({
                key: inputs[key].to(self.device)
            })
        return kwargs_params


_PROMPTS = [
    [{'image': '/path/to/your/picture0.jpeg'},
     {'text': 'Generate the caption in English with grounding:'}],
    [{'image': '/path/to/your/picture1.jpeg'},
     {'text': 'Generate the caption in English with grounding:'}],
    [{'image': '/path/to/your/picture2.jpeg'},
     {'text': 'Generate the caption in English with grounding:'}],
    [{'image': '/path/to/your/picture3.jpeg'},
     {'text': 'Generate the caption in English with grounding:'}],
]


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--model_path', type=str, help="Location of model weights")
    parser.add_argument('--execute_mode', type=str, default="dynamo", choices=["dynamo", "eager"],
                        help="eager or dynamo")
    parser_args = parser.parse_args()
    return parser_args


def run_qwen_vl(model_path, **kwargs):
    model_runner = ModelRunner(model_path, **kwargs)
    # 表示在图模式下开启二进制编译，提高图模式下编译阶段性能
    torch.npu.set_compile_mode(jit_compile=False)
    model_runner.init_model()
    # warmup
    model_runner.model_generate(_PROMPTS, **kwargs)
    # generate perf data
    model_runner.model_generate(_PROMPTS, decode_out=True, **kwargs)


if __name__ == "__main__":
    args = parse_args()
    input_max_len = 1024  # 输入padding的长度
    max_new_tokens = 1024  # 最大输出token个数
    max_position_embeddings = (input_max_len + max_new_tokens)  # 用于申请kv_cache时指定seq_len长度
    config = {
        "dtype": torch.bfloat16,
        "input_max_len": input_max_len,
        "max_new_tokens": max_new_tokens,
        "max_position_embeddings": max_position_embeddings,
    }
    os.environ["EXE_MODE"] = args.execute_mode
    run_qwen_vl(args.model_path, **config)
    logging.info("model run success")
