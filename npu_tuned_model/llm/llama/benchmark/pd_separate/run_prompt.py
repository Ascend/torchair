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
import stat
import time
import argparse
import json
from typing import Dict
import pickle
import logging
import torch
import torch_npu
import torchair
from llm_datadist import LLMDataDist, LLMRole, LLMConfig, ModelRunner, LLMReq, CacheDesc, DataType, CacheKey, KvCache
from llm_inference import SeparateDeployModelRunner

_INVALID_ID = 2 ** 64 - 1
# 需要根据实际的全量机器的device ip进行调整
# 可通过for i in {0..7}; do hccn_tool -i $i -ip -g; done获取机器的device ip信息
_LISTEN_IP_INFO = ["1.1.1.0:26000", "1.1.1.1:26000", "1.1.1.2:26000", "1.1.1.3:26000", "1.1.1.4:26000", "1.1.1.5:26000",
                   "1.1.1.6:26000", "1.1.1.7:26000"]

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)


class LlmPromptRunner(SeparateDeployModelRunner):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path=model_path, **kwargs)

    def prepare_prompt_inputs(self, prompts, **kwargs):
        inputs_tokenizer = self.tokenizer(prompts,
                                          return_tensors="pt",  # 返回pytorch tensor
                                          truncation=True,
                                          padding='max_length',
                                          max_length=kwargs.get("input_max_len", 1024))

        kwargs_params = self._generate_params(inputs_tokenizer)
        return kwargs_params

    @torch.inference_mode()
    def execute_model(self, inputs_tokenizer, **kwargs):
        # 此处不同模型处理输入的逻辑存在差异
        model_inputs = self.model.prepare_inputs_for_generation(input_ids=inputs_tokenizer["input_ids"],
                                                                attention_mask=inputs_tokenizer["attention_mask"],
                                                                **kwargs)
        outputs = self.model_generate(model_inputs)

        return outputs


_PROMPTS = [
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
    parser.add_argument('--execute_mode', type=str, default="dynamo", choices=["dynamo", "eager"],
                        help="eager or dynamo")
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank id")
    parser_args = parser.parse_args()
    return parser_args


# 初始化分离部署的engine
def init_llm_engine(rank_id: int) -> LLMDataDist:
    cluster_id = 0
    engine = LLMDataDist(LLMRole.PROMPT, cluster_id)
    llm_config = LLMConfig()
    llm_config.device_id = rank_id
    # listen_ip_info的ip需要根据实际全量机器的device ip地址进行设置， ip和port需要和增量脚本中remote_ip_info一致
    llm_config.listen_ip_info = _LISTEN_IP_INFO[rank_id]
    ge_options = {
        # 和kv cache的大小有关联，全量需要大于一个请求kv cache的N倍，增量至少需要大于1个请求kv cache
        "ge.flowGraphMemMaxSize": "2589934592",
    }
    llm_config.ge_options = ge_options
    engine.init(llm_config.generate_options())
    return engine


def run_model(cache: KvCache, input_tensors: Dict, runner: LlmPromptRunner, **kwargs):
    kv_tensor_addrs = cache.per_device_tensor_addrs[0]
    kv_tensors = torchair.llm_datadist.create_npu_tensors(cache.cache_desc.shape, torch.float16, kv_tensor_addrs)
    mid = len(kv_tensors) // 2
    k_tensors = kv_tensors[: mid]
    v_tensors = kv_tensors[mid:]
    kv_cache_tensors = list(zip(k_tensors, v_tensors))
    # 此处传递的参数根据不同模型区分全量和增量的入参进行调整
    kwargs["kv_tensors"] = kv_cache_tensors
    outputs = runner.execute_model(input_tensors, **kwargs)
    return outputs


def _allocate_kv_cache(cache_manager, start_id, end_id):
    # 申请kv cache内存。kv shape和num_tensor和实际模型相关，不同模型需要调整。
    # num_tensors是等于layers * 2
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    kv_cache_desc = CacheDesc(num_tensors=160, shape=[4, 2048, 8 // world_size, 128],
                              data_type=DataType.DT_FLOAT16)
    kv_cache_keys = []
    for i in range(start_id, end_id):
        kv_cache_keys.append(CacheKey(prompt_cluster_id=0, req_id=i, model_id=0))
    cache = cache_manager.allocate_cache(kv_cache_desc, kv_cache_keys)
    return cache, kv_cache_keys


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
    flags = os.O_RDWR | os.O_CREAT
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open("prompt.pkl", flags, mode), "wb") as f:
        pickle.dump(_tensor_to_numpy(params), f)


if __name__ == "__main__":
    args = parse_args()
    local_rank = args.local_rank
    prompt_engine = init_llm_engine(local_rank)
    torch.npu.set_device(local_rank)

    config = {
        "dtype": torch.float16,  # 和模型权重目录中config.json里的torch_dtype一致
        "input_max_len": 1024,  # 输入padding的长度
        "max_new_tokens": 1024,  # 最大输出token个数
    }
    os.environ["EXE_MODE"] = args.execute_mode
    model_runner = LlmPromptRunner(args.model_path, **config)
    # 表示开启二进制编译，提高编译阶段性能
    torch.npu.set_compile_mode(jit_compile=False)
    model_runner.init_model()
    # 根据execute_mode选择是否走图模式编译
    model_runner.compile_model()

    kv_cache_manager = prompt_engine.kv_cache_manager

    # warmup
    warmup_kv_cache, warmup_cache_keys = _allocate_kv_cache(kv_cache_manager, 0, 4)
    # _PROMPTS里的个数表示模型执行一次的batch size大小
    inputs = model_runner.prepare_prompt_inputs(_PROMPTS, **config)
    _ = run_model(warmup_kv_cache, inputs, model_runner, **config)
    model_runner.reset_time_statistics()

    # 模型推理，将申请好的kv cache传递给模型，替换原模型中的kv cache tensor
    kv_cache, cache_keys = _allocate_kv_cache(prompt_engine.kv_cache_manager, 4, 8)
    result = run_model(kv_cache, inputs, model_runner, **config)
    avg_time = model_runner.inference_avg_time()
    logging.info(f"Average token cost time: {avg_time} s")
    send_prompt_outputs(inputs, result)
    logging.info("model run success")
    time.sleep(1200)  # 目的是为了保持全量进程不立马退出导致增量建链和拉kv失败，根据业务自行调整

    # 清理kv cache资源
    for allocated_cache in [warmup_kv_cache, kv_cache]:
        kv_cache_manager.deallocate_cache(allocated_cache)
    for cache_keys in (warmup_cache_keys + cache_keys):
        kv_cache_manager.remove_cache_key(cache_keys)
    prompt_engine.finalize()
