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
from typing import List
import torch
import torch_npu
import torchair
from llm_datadist import LLMClusterInfo, LLMDataDist, LLMRole, LLMConfig, ModelRunner, LLMReq
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


# 自定义了一个ModelRunner类，并重写run_model方法，里面调用torch的模型执行接口
class TorchModelRunner(ModelRunner):
    def __init__(self, runner, kv_cache_manager):
        self._model_runner = runner
        self._kv_cache_manager = kv_cache_manager

    def run_model(self, kv_cache, input_tensors: List, **kwargs) -> List:
        kv_tensor_addrs = kv_cache.per_device_tensor_addrs[0]
        kv_tensors = torchair.llm_datadist.create_npu_tensors(kv_cache.cache_desc.shape, torch.float16, kv_tensor_addrs)
        mid = len(kv_tensors) // 2
        k_tensors = kv_tensors[: mid]
        v_tensors = kv_tensors[mid:]
        kv_cache_tensors = list(zip(k_tensors, v_tensors))
        # 此处传递的参数根据不同模型区分全量和增量的入参进行调整
        kwargs["past_key_values"] = kv_cache_tensors
        # input_tensors[0]对应predict接口入参inputs
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
    with os.fdopen(os.open("prompt.pkl", os.O_RDONLY, stat.S_IRUSR), "rb") as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def _init_model_options():
    # 构造kv shape的dtype和shape，需要根据网络的不同情况进行调整
    num_layers = 80
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    kv_shape = f"4,2048,{8 // world_size},128"
    shapes_str = ";".join([kv_shape] * num_layers * 2)
    dtypes_str = ";".join(['1'] * num_layers * 2)
    model_options = {
        "llm.RefInputShapes": shapes_str,
        "llm.RefInputDtypes": dtypes_str
    }
    return model_options


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
    # 表示在图模式下开启二进制编译，提高图模式下编译阶段性能
    torch.npu.set_compile_mode(jit_compile=False)
    model_runner.init_model()
    # 走图模式需要进行图编译
    model_runner.compile_model()

    batch_size = 4  # bs大小取决于模型给的输入的bs大小
    llm_model = decoder_engine.add_model(_init_model_options(),
                                         TorchModelRunner(model_runner, decoder_engine.kv_cache_manager))
    llm_reqs = _init_llm_req(llm_model, 0, 4, batch_size)
    step_inputs = _numpy_to_tensor(recv_inputs_from_prompt())
    # warmup
    inputs = model_runner.prepare_decoder_inputs(step_inputs, **config)
    _ = llm_model.predict(llm_reqs, [inputs], **config)
    model_runner.reset_time_statistics()

    # 增量迭代推理，可以自己控制请求的加入和退出，新请求加入前需要pull_kv和merge_kv
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
    decoder_engine.finalize()
    logging.info("model run success")
