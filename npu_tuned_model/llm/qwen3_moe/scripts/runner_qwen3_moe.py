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
from functools import wraps
import numpy as np
import torch
import torch_npu

from engine.model_runner import ModelRunner
from models.modeling_qwen3_moe import Qwen3MoeForCausalLM

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)

torch.manual_seed(42)
torch.npu.manual_seed_all(42)


def override(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def get_init_attn_mask(mask_length, device, valid_len=None):
    share_mask_tril = ~torch.tril(
        torch.ones((mask_length, mask_length),
                   dtype=torch.bool, device=device))
    if valid_len is not None:
        share_mask_tril[-valid_len:, :] = torch.zeros(valid_len, mask_length)
    return share_mask_tril


def get_decode_mask(mask_length, device, position):
    decode_mask = torch.zeros((1, mask_length), device=device)
    decode_mask[0, :position] = 1
    return decode_mask


class Qwen3MoeRunner(ModelRunner):
    def __init__(self, model_path, execute_mode, **kwargs):
        super().__init__(model_path, execute_mode, **kwargs)
        self.enable_mla = kwargs.get("enable_mla", 0)
        self.no_ckpt = int(os.getenv("NO_CKPT", "0"))
        self.enable_mix = int(os.getenv("ENABLE_MIX", "0"))
        if self.enable_mix:
            self.attn_dp_size = int(os.getenv("ATTN_DP_SIZE", "0"))
        else:
            self.attn_dp_size = 1
    
    def init_model(self):
        if not self.no_ckpt:
            self.use_pretrained_model = True
            config = None
        else:
            self.use_pretrained_model = False
            from models.configuration_qwen3_moe import Qwen3MoeConfig as config
        super().init_model(Qwen3MoeForCausalLM, config)
    
    @override        
    def mark_inputs(self, model_inputs):
        if self.execute_mode == "dynamo":
            input_ids = model_inputs.get("input_ids")
            kv_len = model_inputs.get("kv_len")
            attention_mask = model_inputs.get("attention_mask")
            position_ids = model_inputs.get("position_ids")
            past_key_values = model_inputs.get("past_key_values")

            # prefill with dynamic sequence length, decode with static sequence length
            torch._dynamo.mark_static(kv_len)
            for item in past_key_values:
                for sub_item in item:
                    torch._dynamo.mark_static(sub_item)
            
            torch._dynamo.mark_static(input_ids)
            if attention_mask is not None:
                torch._dynamo.mark_static(attention_mask)
            torch._dynamo.mark_static(position_ids)

    @override
    def model_input_prepare(self, input_dict):
        input_ids = input_dict.get("input_ids")
        attention_mask = input_dict.get("attention_mask")
        past_key_values = input_dict.get("past_key_values")
        is_prefill = input_dict.get("is_prefill")
        kv_len = input_dict.get("kv_len")
        share_mask_tril = input_dict.get("share_mask_tril")
        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            is_prefill=is_prefill,
            kv_len=kv_len,
            input_lens=input_dict.get("input_lens"),
            share_mask_tril=share_mask_tril,
            world_size=self.world_size)
        return model_inputs

    @override
    def model_output_process(self, model_inputs, outputs, input_dict):
        next_batch = self.batch_size if input_dict["is_prefill"] else 1
        next_batch_dp = next_batch // self.attn_dp_size if input_dict["is_prefill"] else 1
        input_dict['is_prefill'] = False
        input_dict['input_lens'] = input_dict['input_lens'] + 1

        kv_len = torch.max(model_inputs.get("position_ids"), axis=1)[0] + 1
        input_dict['kv_len'] = ModelRunner.repeat_batch(kv_len, next_batch_dp)

        logits = outputs
        past_key_values = model_inputs.get("past_key_values")
        past_key_values_batch = ()
        for past_key_values_layer_i in past_key_values:
            cache_new_i = ()
            for cache_j in past_key_values_layer_i:
                cache_j_new = ModelRunner.repeat_batch(cache_j, next_batch_dp)
                cache_new_i += (cache_j_new, )
            past_key_values_batch += (cache_new_i, )
        input_dict["past_key_values"] = past_key_values_batch
        
        attention_mask = None

        share_mask_tril = get_decode_mask(mask_length=self.max_position_embeddings,
                                            device=self.device,
                                            position=input_dict["input_lens"])
        share_mask_tril = share_mask_tril[None, None, ...]

        input_dict['attention_mask'] = attention_mask
        input_dict['share_mask_tril'] = ModelRunner.repeat_batch(share_mask_tril, self.batch_size)

        next_tokens = torch.argmax(logits, dim=-1)[:, -1:]
        input_dict['input_ids'] = ModelRunner.repeat_batch(next_tokens, next_batch)
        input_dict['generate_ids'] = ModelRunner.repeat_batch(
                                            torch.cat([input_dict['generate_ids'], next_tokens], dim=-1),
                                            next_batch
                                        )

    @override
    def model_generate(self, prompts, warm_up=False, **kwargs):
        calling_func = {"default": self.tokenizer, "chat": self.tokenizer.apply_chat_template}
        kwargs = {
            "return_tensors": "pt", "truncation": True, "padding": "max_length", "max_length": self.input_max_len
        }
        if self.tokenizer_mode == "chat":
            chat_kwargs = {"add_generation_prompt": True, "return_dict": True}
            kwargs.update(chat_kwargs)
        tokenizer = calling_func.get(self.tokenizer_mode, self.tokenizer)
        inputs = tokenizer(prompts, **kwargs).to(self.device)

        # get init input_dict
        share_mask_tril = get_init_attn_mask(2048, self.device)
        
        input_lens = copy.deepcopy(inputs.input_ids.size()[1])
        logging.info("Prompt lens is : %d", input_lens)
        input_dict = {
            "input_ids": inputs.input_ids, "generate_ids": inputs.input_ids,
            "input_lens": input_lens, "kv_len": None,
            "past_key_values": None, "attention_mask": inputs.attention_mask, "share_mask_tril": share_mask_tril,
            "is_prefill": True,
        }

        generate_tokens = 0
        cnt = 0
        while True:
            jump_flag = self.get_jump_flag(cnt, warm_up, generate_tokens)
            if jump_flag:
                break

            model_inputs = self.model_input_prepare(input_dict)
            outputs = self.model_inference(model_inputs, warm_up=warm_up)
            self.model_output_process(model_inputs, outputs, input_dict)
            generate_tokens += 1
            cnt += 1

        generate_ids = input_dict["generate_ids"][0:1, input_lens:].clip(0, self.model.config.vocab_size - 1)
        res = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

        if isinstance(res, list):
            for answer in res:
                logging.info("Inference decode result: \n%s", answer)
        else:
            logging.info("Inference decode result: \n%s", res)
        return res

    def get_jump_flag(self, cnt, warm_up, generate_tokens):
        default_decode_dump = 2
        # warm up only perform for 5 times(decode)
        jump_flag_warm = warm_up and cnt >= default_decode_dump
        # do not generate after max_token
        jump_flag_oversize = generate_tokens >= self.max_new_tokens
        jump_flag = jump_flag_oversize or jump_flag_warm
        return jump_flag

