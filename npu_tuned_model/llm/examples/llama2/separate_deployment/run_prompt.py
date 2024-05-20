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
import json
from typing import List
import pickle
import logging
import torch
import llm_link_torch
from llm_engine import *
from llm_engine.v2.llm_engine_v2 import LLMEngine
from runner.separate_deployment.llm_prompt import LlmPromptRunner

_INVALID_ID = 2 ** 64 - 1

prompts = [
    "用一句话描述地球为什么是独一无二的。",
    "给出一段对话，使用合适的语气和回答方式继续对话。\n对话：\nA：你今天看起来很高兴，发生了什么好事？\nB：是的，我刚刚得到一份来自"
    "梅西银行的工作通知书。\nA：哇，恭喜你！你打算什么时候开始工作？\nB：下个月开始，所以我现在正为这份工作做准备。",
    "基于以下提示填写以下句子的空格。\n提示：\n- 提供多种现实世界的场景\n- 空格应填写一个形容词或一个形容词短语\n句子:\n______出去"
    "享受户外活动，包括在公园里散步，穿过树林或在海岸边散步。",
    "请生成一个新闻标题，描述一场正在发生的大型自然灾害。",
]


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--model_path', type=str, help="Location of model weights")
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank id")
    parser_args = parser.parse_args()
    return parser_args


def init_llm_engine(rank_id: int) -> LLMEngine:
    cluster_id = 0
    engine = LLMEngine(LLMRole.PROMPT, cluster_id)
    # listen_ip_info的ip需要根据实际全量机器的device ip地址进行设置， ip和port需要和增量脚本中remote_ip_info一致
    cluster_info_str = {"cluster_id": 0,
                        "logic_device_id":
                            ["0:0:0:0", "0:0:1:0", "0:0:2:0", "0:0:3:0", "0:0:4:0", "0:0:5:0", "0:0:6:0", "0:0:7:0"],
                        "listen_ip_info": [
                            {"ip": 1865362717, "port": 26000}, {"ip": 444083485, "port": 26000},
                            {"ip": 795946269, "port": 26000}, {"ip": 995044637, "port": 26000},
                            {"ip": 2456169757, "port": 26000}, {"ip": 1158098205, "port": 26000},
                            {"ip": 3060739357, "port": 26000}, {"ip": 3205508381, "port": 26000}
                        ]}
    options = {
        "llm.ClusterInfo": json.dumps(cluster_info_str),
        "ge.exec.rankId": str(rank_id),
        "ge.flowGraphMemMaxSize": "2589934592",
    }
    engine.init(options)
    return engine


# 自定义了一个ModelRunner类，并重写run_model方法，里面调用torch的模型执行接口
class TorchModelRunner(ModelRunner):
    def __init__(self, runner, kv_cache_manager):
        self._model_runner = runner
        self._kv_cache_manager = kv_cache_manager

    def run_model(self, kv_cache, input_tensors: List, **kwargs) -> List:
        kv_tensor_addrs = kv_cache.per_device_tensor_addrs[0]
        kv_tensors = llm_link_torch.create_npu_tensors(kv_cache.cache_desc.shape, torch.float16, kv_tensor_addrs)
        mid = len(kv_tensors) // 2
        k_tensors = kv_tensors[: mid]
        v_tensors = kv_tensors[mid:]
        kv_cache_tensors = list(zip(k_tensors, v_tensors))
        # 此处传递的参数根据不同模型区分全量和增量的入参进行调整
        kwargs["kv_tensors"] = kv_cache_tensors
        outputs = self._model_runner.execute_model(input_tensors[0], **kwargs)
        return outputs


def _llm_req(req_id: int, prefix_id: int = -1, prompt_length: int = 256) -> LLMReq:
    llm_req = LLMReq()
    llm_req.req_id = req_id
    llm_req.prefix_id = prefix_id
    llm_req.prompt_length = prompt_length
    return llm_req


def _init_llm_req(req_num):
    reqs = []
    for i in range(req_num):
        llm_req = _llm_req(i, _INVALID_ID, 1024)
        reqs.append(llm_req)
    return reqs


# 此处是为了将全量的输出以numpy保存做了tensor到numpy的转换
def _tensor_to_numpy(params):
    res = {}
    for key in params.keys():
        res.update({
            key: params[key].cpu().numpy()
        })
    return res


# 此处是将全量输出发送给增量，用例是以文件方式传递数据，可根据实际业务创建进行更改
def send_prompt_outputs(model_inputs, model_outputs):
    params = {}
    params["generate_ids"] = model_outputs[:, None]
    params["input_ids"] = model_inputs["input_ids"]
    params["attention_mask"] = model_inputs["attention_mask"]
    with open("prompt.pkl", "wb") as f:
        pickle.dump(_tensor_to_numpy(params), f)


if __name__ == "__main__":
    args = parse_args()
    local_rank = args.local_rank
    prompt_engine = init_llm_engine(local_rank)
    torch.npu.set_device(local_rank)

    config = {
        "input_padding": True,
        "dtype": torch.float16,
        "input_max_len": 1024,
        "max_new_tokens": 1024,
    }

    model_runner = LlmPromptRunner("llama2", args.model_path, **config)
    model_runner.execute_mode = "dynamo"
    model_runner.set_npu_config(jit_compile=False)
    model_runner.init_model()
    model_runner.compile_model()

    num_layers = 80
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    kv_shape = f"4,2048,{8 // world_size},128"
    shapes_str = ";".join([kv_shape] * num_layers * 2)
    dtypes_str = ";".join(['1'] * num_layers * 2)
    model_options = {
        "llm.RefInputShapes": shapes_str,
        "llm.RefInputDtypes": dtypes_str
    }
    llm_model = prompt_engine.add_model(model_options, TorchModelRunner(model_runner, prompt_engine.kv_cache_manager))

    inputs = model_runner.prepare_prompt_inputs(prompts, **config)
    llm_reqs = _init_llm_req(8)
    # warmup
    _ = llm_model.predict(llm_reqs[:4], [inputs], **config)
    model_runner.reset_time_statistics()
    # inference
    result = llm_model.predict(llm_reqs[4:], [inputs], **config)
    avg_time = model_runner.inference_avg_time()
    logging.info(f"Average token cost time: {avg_time} s")
    send_prompt_outputs(inputs, result)
    logging.info("model run success")
    time.sleep(1200)  # 目的是为了保持全量进程不立马退出导致增量建链和拉kv失败，根据业务自行调整
    for req in llm_reqs:
        prompt_engine.complete_request(req)