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
import argparse
import logging
import json
import pickle
from typing import Dict
import torch
import torch_npu
import torchair
from llm_datadist import LLMClusterInfo, LLMDataDist, LLMRole, LLMConfig, ModelRunner, CacheKey, CacheDesc, DataType, \
    KvCache
from llm_inference import SeparateDeployModelRunner

_INVALID_ID = 2 ** 64 - 1
# 此处ip信息需要根据实际的device ip进行更改
# 可通过for i in {0..7}; do hccn_tool -i $i -ip -g; done获取机器的device ip信息
_LOCAL_IP_INFOS = ["1.1.1.0", "1.1.1.1", "1.1.1.2", "1.1.1.3", "1.1.1.4", "1.1.1.5", "1.1.1.6", "1.1.1.7"]
_REMOTE_IP_INFOS = ["2.2.2.0", "2.2.2.1", "2.2.2.2", "2.2.2.3", "2.2.2.4", "2.2.2.5", "2.2.2.6", "2.2.2.7"]

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)


class LlmDecoderRunner(SeparateDeployModelRunner):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path=model_path, **kwargs)

    # 增量的首次输入来自全量，并在后续迭代过程中更新，更新的逻辑不同模型有差异，以具体模型为准
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
    def execute_model(self, params, **kwargs):
        model_inputs = self.model.prepare_inputs_for_generation(input_ids=params["input_ids"],
                                                                attention_mask=params["attention_mask"],
                                                                **kwargs)
        outputs = self.model_generate(model_inputs)
        return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--model_path', type=str, help="Location of model weights")
    parser.add_argument('--execute_mode', type=str, default="dynamo", choices=["dynamo", "eager"],
                        help="eager or dynamo")
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank id")
    parser_args = parser.parse_args()
    return parser_args


def init_clusters_info(rank_id):
    cluster = LLMClusterInfo()
    cluster.remote_cluster_id = 0
    # 此处local_ip和remote_ip需要一一对应
    # local_ip_info的port可以随便填，remote_ip_info的port需要指定可用port并且和全量的listen_ip_info设置一致
    # 对于一个增量cluster，支持1v N全量
    cluster.append_local_ip_info(_LOCAL_IP_INFOS[rank_id], 26000)
    cluster.append_remote_ip_info(_REMOTE_IP_INFOS[rank_id], 26000)
    clusters = [cluster]
    return clusters


def init_llm_engine(rank_id: int) -> LLMDataDist:
    cluster_id = 0
    engine = LLMDataDist(LLMRole.DECODER, cluster_id)
    llm_config = LLMConfig()
    llm_config.device_id = rank_id
    ge_options = {
        # 和kv cache的大小有关联，全量需要大于一个请求kv cache的N倍，增量至少需要大于1个请求kv cache
        "ge.flowGraphMemMaxSize": "2589934592",
    }
    llm_config.ge_options = ge_options
    engine.init(llm_config.generate_options())
    return engine


def run_model(cache: KvCache, input_tensors: Dict, runner: LlmDecoderRunner, **kwargs):
    kv_tensor_addrs = cache.per_device_tensor_addrs[0]
    kv_tensors = torchair.llm_datadist.create_npu_tensors(cache.cache_desc.shape, torch.float16, kv_tensor_addrs)
    mid = len(kv_tensors) // 2
    k_tensors = kv_tensors[: mid]
    v_tensors = kv_tensors[mid:]
    kv_cache_tensors = list(zip(k_tensors, v_tensors))
    # 此处传递的参数根据不同模型区分全量和增量的入参进行调整
    kwargs["past_key_values"] = kv_cache_tensors
    outputs = runner.execute_model(input_tensors, **kwargs)
    return outputs


# 申请增量kv cache，并且从全量拉取kv，并且将kv copy到指定batch位置
def _allocate_kv_cache(cache_manager):
    # 申请kv cache内存。kv shape和num_tensor和实际模型相关，不同模型需要调整。
    # num_tensors是等于layers * 2
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    kv_cache_desc = CacheDesc(num_tensors=160, shape=[4, 2048, 8 // world_size, 128],
                              data_type=DataType.DT_FLOAT16)
    cache = cache_manager.allocate_cache(kv_cache_desc)
    return cache


def _init_kv_cache(cache_manager, cache, start_id, end_id, bs):
    for index in range(start_id, end_id):
        prompt_cache_key = CacheKey(prompt_cluster_id=0, req_id=index, model_id=0)
        cache_manager.pull_cache(prompt_cache_key, cache, index % bs)


# 此函数是因为接收的全量数据是numpy，需要转成tensor后作为模型输入
def _numpy_to_tensor(params):
    res = {}
    for key in params.keys():
        res.update({
            key: torch.from_numpy(params[key])
        })
    return res


# 此函数是接收来自全量的输出作为增量的输入，此处仅通过文件的形式进行数据传递，可替换为实际业务场景数据传递方式
def recv_inputs_from_prompt():
    with os.fdopen(os.open("prompt.pkl", os.O_RDONLY, stat.S_IRUSR), "rb") as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


if __name__ == "__main__":
    args = parse_args()
    local_rank = args.local_rank
    decoder_engine = init_llm_engine(local_rank)
    torch.npu.set_device(local_rank)
    clusters_info = init_clusters_info(local_rank)
    decoder_engine.link_clusters(clusters_info)

    max_new_tokens = 1024
    config = {
        "dtype": torch.float16,  # 和模型权重目录中config.json里的torch_dtype一致
        "input_max_len": 1024,  # 输入padding的长度
        "max_new_tokens": max_new_tokens,  # 最大输出token个数
    }

    os.environ["EXE_MODE"] = args.execute_mode
    model_runner = LlmDecoderRunner(args.model_path, **config)
    # 表示开启二进制编译，提高编译阶段性能
    torch.npu.set_compile_mode(jit_compile=False)
    model_runner.init_model()
    # 根据execute_mode选择是否走图模式编译
    model_runner.compile_model()

    kv_cache_manager = decoder_engine.kv_cache_manager
    kv_cache = _allocate_kv_cache(kv_cache_manager)

    batch_size = 4  # bs大小取决于模型给的输入的bs大小
    # warmup，将申请好的kv cache传递给模型，替换原模型中的kv cache tensor
    _init_kv_cache(kv_cache_manager, kv_cache, 0, 4, batch_size)
    step_inputs = _numpy_to_tensor(recv_inputs_from_prompt())
    inputs = model_runner.prepare_decoder_inputs(step_inputs, **config)
    _ = run_model(kv_cache, inputs, model_runner, **config)
    model_runner.reset_time_statistics()

    # 增量迭代推理，可以自己控制请求的加入和退出，新请求加入前需要pull_cache，若想pull_cache和nn推理流水，可以结合copy_cache使用
    _init_kv_cache(kv_cache_manager, kv_cache, 4, 8, batch_size)
    input_len = len(step_inputs["input_ids"][0])
    for i in range(max_new_tokens):
        inputs = model_runner.prepare_decoder_inputs(step_inputs, **config)
        result = run_model(kv_cache, inputs, model_runner, **config)
        if i == max_new_tokens - 1:
            step_inputs["input_ids"] = torch.cat([inputs["input_ids"], result[:, None]], dim=-1)
            break
        step_inputs["input_ids"] = inputs["input_ids"]
        step_inputs["generate_ids"] = result[:, None]
        step_inputs["attention_mask"] = inputs["attention_mask"]
    avg_time = model_runner.inference_avg_time()
    logging.info(f"Average token cost time: {avg_time} s")
    output_len = len(step_inputs["input_ids"][0])
    output_tokens = model_runner.process_output(step_inputs["input_ids"][:, (output_len - input_len):])
    if local_rank == 0:
        if isinstance(output_tokens, list):
            for answer in output_tokens:
                logging.info(f"Inference decode result: \n{answer}")
        else:
            logging.info(f"Inference decode result: \n{output_tokens}")

    kv_cache_manager.deallocate_cache(kv_cache)
    decoder_engine.unlink_clusters(clusters_info)
    decoder_engine.finalize()
    logging.info("model run success")
