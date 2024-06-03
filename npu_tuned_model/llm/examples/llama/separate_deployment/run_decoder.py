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
import argparse
import logging
import json
import pickle
from typing import List
import torch
import llm_link_torch
from llm_engine import *
from llm_engine.v2.llm_engine_v2 import LLMEngine
from runner.separate_deployment.llm_decoder import LlmDecoderRunner

_INVALID_ID = 2 ** 64 - 1


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--model_path', type=str, help="Location of model weights")
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank id")
    parser_args = parser.parse_args()
    return parser_args


def init_clusters_info():
    cluster = LLMClusterInfo()
    cluster.remote_cluster_id = 0
    # 此处需要更改为使用机器的大端device ip的int值, local_ip和remote_ip的顺序一一对应
    # local_ip_info的port可以随便填，remote_ip_info的port需要指定可用port并且和全量的listen_ip_info设置一致
    cluster.append_local_ip_info(3517263133, 26000)
    cluster.append_local_ip_info(907029789, 26000)
    cluster.append_local_ip_info(641871133, 26000)
    cluster.append_local_ip_info(2503355677, 26000)
    cluster.append_local_ip_info(1069034781, 26000)
    cluster.append_local_ip_info(1297755421, 26000)
    cluster.append_local_ip_info(2455973149, 26000)
    cluster.append_local_ip_info(796208413, 26000)
    cluster.append_remote_ip_info(1865362717, 26000)
    cluster.append_remote_ip_info(444083485, 26000)
    cluster.append_remote_ip_info(795946269, 26000)
    cluster.append_remote_ip_info(995044637, 26000)
    cluster.append_remote_ip_info(2456169757, 26000)
    cluster.append_remote_ip_info(1158098205, 26000)
    cluster.append_remote_ip_info(3060739357, 26000)
    cluster.append_remote_ip_info(3205508381, 26000)
    clusters = [cluster]
    return clusters


def init_llm_engine(rank_id: int) -> LLMEngine:
    cluster_id = 0
    engine = LLMEngine(LLMRole.DECODER, cluster_id)
    cluster_info_str = {"cluster_id": 0,
                        "logic_device_id":
                            ["0:0:0:0", "0:0:1:0", "0:0:2:0", "0:0:3:0", "0:0:4:0", "0:0:5:0", "0:0:6:0", "0:0:7:0"]}
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
        kwargs["past_key_values"] = kv_cache_tensors
        outputs = self._model_runner.execute_model(input_tensors[0], **kwargs)
        return outputs


def _llm_req(req_id: int, prefix_id: int = -1, prompt_length: int = 256) -> LLMReq:
    llm_req = LLMReq()
    llm_req.req_id = req_id
    llm_req.prefix_id = prefix_id
    llm_req.prompt_length = prompt_length
    return llm_req


# 创建对应的分离部署请求，并且从全量拉取kv，并且将kv copy到指定batch位置
def _init_llm_req(model, start_id, end_id, bs):
    reqs = []
    for index in range(start_id, end_id):
        llm_req = _llm_req(index, _INVALID_ID, 1024)
        model.pull_kv(llm_req)
        model.merge_kv(llm_req, index % bs)
        reqs.append(llm_req)
    return reqs


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
    with open("prompt.pkl", "rb") as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


if __name__ == "__main__":
    args = parse_args()
    local_rank = args.local_rank
    decoder_engine = init_llm_engine(local_rank)
    torch.npu.set_device(local_rank)
    clusters_info = init_clusters_info()
    decoder_engine.link_clusters(clusters_info)

    max_new_tokens = 1024
    config = {
        "input_padding": True,
        "dtype": torch.float16,
        "input_max_len": 1024,
        "max_new_tokens": max_new_tokens,
    }

    model_runner = LlmDecoderRunner("llama2", args.model_path, **config)
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
    batch_size = 4  # bs大小取决于模型给的输入的bs大小
    llm_model = decoder_engine.add_model(model_options, TorchModelRunner(model_runner, decoder_engine.kv_cache_manager))
    llm_reqs = _init_llm_req(llm_model, 0, 4, batch_size)
    step_inputs = _numpy_to_tensor(recv_inputs_from_prompt())
    # warmup
    inputs = model_runner.prepare_decoder_inputs(step_inputs, **config)
    _ = llm_model.predict(llm_reqs, [inputs], **config)
    model_runner.reset_time_statistics()

    llm_reqs = _init_llm_req(llm_model, 4, 8, batch_size)
    input_len = len(step_inputs["input_ids"][0])
    for i in range(max_new_tokens):
        inputs = model_runner.prepare_decoder_inputs(step_inputs, **config)
        result = llm_model.predict(llm_reqs, [inputs], **config)
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

    decoder_engine.unlink_clusters(clusters_info)
    logging.info("model run success")
