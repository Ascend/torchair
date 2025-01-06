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
import torch_npu
import torchair as tng
config = tng.CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)

class ModelRunner:
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.quant_mode = kwargs.get("quant_mode", "")
        self.model = None
        self.device = None
        self.local_rank = 0

    def init_model(self):
        logging.info("Set execution using npu index: %s", self.local_rank)
        self.device = torch.device("%s:%s" % ("npu", self.local_rank))
        torch.npu.set_device(self.device)

        logging.info("Try to load pretrained model in path: %s", self.model_path)
        if self.quant_mode == "":
            from model import Qwen2VLForConditionalGeneration
        else:
            raise ValueError(f"quant mode:{self.quant_mode} is not support currently.")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map = "balanced",
        )
        self.model.to(self.device)
        logging.info("The final model structure is: \n %s", self.model)

    def model_generate(self, prompts, decode_out=False, **kwargs):
        # Preparation for inference
        processor = AutoProcessor.from_pretrained(self.model_path)
        text = processor.apply_chat_template(
            prompts, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(prompts)
        decode_st = time.time()
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        logging.info("Processor decode success, time cost: %.2fs", time.time() - decode_st)

        kwargs_params = self._generate_params(inputs, kwargs.get("max_new_tokens", 128))
        generate_st = time.time()
        generated_ids = self.model.generate(**kwargs_params)
        logging.info("Model execute success, time cost: %.2fs", time.time() - generate_st)

        if not decode_out:
            return
        input_tokens = len(inputs.input_ids[0])
        output_tokens = len(generated_ids[0])
        new_tokens = output_tokens - input_tokens

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        res = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

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

# prompts的size大小决定了模型执行时的batch size大小
_PROMPTS = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "video.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--model_path', type=str, help="Location of model weights")
    parser.add_argument('--execute_mode', type=str, default="dynamo", choices=["dynamo", "eager"],
                        help="eager or dynamo")
    parser.add_argument('--quant_mode', type=str, default="", choices=["", "a8w8c8"], 
                        help="set quant mode")
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank id")
    parser_args = parser.parse_args()
    return parser_args


def run_qwen2_vl(model_path, **kwargs):
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
    config = {
        "dtype": torch.bfloat16,  # 和模型权重目录中config.json里的torch_dtype一致
        "input_max_len": 1024,  # 输入padding的长度
        "max_new_tokens": 128,  # 最大输出token个数
        "quant_mode": f"{args.quant_mode}", # 量化类型，默认为空，不做量化操作
    }
    os.environ["EXE_MODE"] = args.execute_mode
    run_qwen2_vl(args.model_path, **config)
    logging.info("model run success")
