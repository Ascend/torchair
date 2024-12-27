

# 分离部署迁移指南

本章节主要介绍如何将一个pytorch的大模型脚本迁移为可以全量和增量分离部署的脚本

# 快速使用

模型结构使用的是已经改造后的modeling_llama.py

**搭建环境**

```shell
# arm环境搭建示例
conda create -n test python=3.9
conda activate test

# 根据CANN安装指南安装固件/驱动/cann包

# 安装 torch 和 torch_npu
pip3 install torch-2.1.0-cp39-cp39m-manylinux2014_aarch64.whl
pip3 install torch_npu-2.1.0*-cp39-cp39m-linux_aarch64.whl
pip3 install apex-0.1_ascend*-cp39-cp39m-linux_aarch64.whl

git clone https://gitee.com/ascend/torchair.git
cd torchair/npu_tuned_model/llm
pip3 install -r requirement.txt

# llama2
pip3 install transformers==4.31.0
# llama3
pip3 install transformers==4.40.0
```

**设置环境变量**

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/your/torchair/npu_tuned_model/llm/llama
cann_path=/usr/local/Ascend # 昇腾cann包安装目录
source ${cann_path}/latest/bin/setenv.bash
export ASCEND_HOME_PATH=${cann_path}/latest
```

**qkv权重融合**

```shell
model_path=xxx/llama2-70b # 下载的权重和模型信息
python3 merge_qkv_weight.py --model_path=${model_path} --tp_size=8 --output_path=xxx/llama-70b_qkv
```

**将替换了mc2融合算子的LinearAllreduce替换deepspeed原生的LinearAllreduce**

将benchmark/deepspeed/mc2_adapter.py的LinearAllreduce整个类拷贝替换原生deepspeed的deepspeed/module_inject/layers.py中的LinearAllreduce类，并且import torch_npu

**注：上述操作分别在全量和增量机器上完成后，**

分别查询全量和增量机器的device_ip信息

```shell
for i in {0..7}; do hccn_tool -i $i -ip -g; done
```

注：出现hccn_tool命令找不到的，可在cann包安装目录下搜索hccn_tool，找到可执行文件执行

更改全量脚本中**_LISTEN_IP_INFO**为实际机器device ip信息，拉起全量的执行脚本，可通过--execute_mode参数选择单算子或者图模式

```shell
deepspeed --num_gpus=8 benchmark/pd_separate/run_prompt.py --model_path=xxx/llama2-70b_qkv
```

将全量执行完目录下生成的prompt.pkl文件拷贝到增量执行目录下，更改增加脚本中的**_LOCAL_IP_INFOS**和**_REMOTE_IP_INFOS**为实际机器device ip信息，拉起增量的执行脚本，可通过--execute_mode参数选择单算子或者图模式

**_LOCAL_IP_INFOS**是增量机器的device ip信息，**_REMOTE_IP_INFOS**是全量机器的device ip信息

```shell
deepspeed --num_gpus=8 benchmark/pd_separate/run_decoder.py --model_path=xxx/llama2-70b_qkv
```

**打点位置**

在llm_inference.py的model_generate函数中start和end

# 迁移步骤

- 准备一个**能够成功执行**推理的大模型脚本。

- 查看脚本执行推理的代码**是否是调用的集成接口**。什么是集成接口，就是调用一次可以完成整个句子长度推理的接口。transformers提供的generate接口示例：

  ```python
  # 参考llm/llama/benchmark/npu/benchmark_llama.py的ModelRunner.model_generate函数
  inputs = self.tokenizer(prompts,
                          return_tensors="pt",  # 返回pytorch tensor
                          truncation=True,
                          padding='max_length',
                          max_length=kwargs.get("input_max_len", 1024))

  kwargs_params = self._generate_params(inputs, kwargs.get("max_new_tokens", 1024))
  start_time = time.time()
  generate_ids = self.model.generate(**kwargs_params)
  elapse = time.time() - start_time
  logging.info("Model execute success, time cost: %.2fs", elapse)
  ```

- 如果是集成接口，那么需要做的**调整脚本执行推理的代码，修改为一次调用只会推理一次**。pytorch主要修改为直接调用模型定义的forward函数。一次调用只做一次推理的示例：

  ```python
  # 参考llm/llama/benchmark/pd_separate/llm_inference.py的SeparateDeployModelRunner.model_generate函数
  torch.npu.synchronize()
  outputs = self.model(**model_inputs)
  torch.npu.synchronize()
  next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)
  ```

- 调整成一次调用只会推理一次后，需要**保证能够和使用集成接口精度和性能一致**

- 上述做完后，根据模型结构定义的代码如何区分全量和增量执行的，将**执行脚本拆分为全量执行脚本和增量执行脚本**。大模型脚本多以seq_length和kv cache区分全量和增量执行。

  以llama2为例，llama2中通过past_key_values这个输入是否是None，区分了全量和增量。当past_key_values为None时是全量推理，非None时是增量推理。示例代码：

  ```python
  # 参考llm/llama/modeling_llama.py的LlamaForCausalLM.prepare_inputs_for_generation函数
  # 根据past_key_values不一样，会导致模型的输入变化。
  if past_key_values:
      input_ids = input_ids[:, -1:]
  input_ids = input_ids.clone()  # 添加clone目的是为了保证fx图上input_ids不变化
  position_ids = kwargs.get("position_ids", None)
  if attention_mask is not None and position_ids is None:
      # create position_ids on the fly for batch generation
      position_ids = attention_mask.long().cumsum(-1) - 1
      position_ids.masked_fill_(attention_mask == 0, 1)
      if past_key_values:
          position_ids = position_ids[:, -1].unsqueeze(-1)
  ```

- 拆分完的脚本有各自的**输入预处理，模型执行和输出后处理**。此处如果是集成接口，需要根据集成接口内部对于输入输出的处理进行调整。

  **全量的输入预处理和模型执行参考**[run_prompt.py](./run_prompt.py)的LlmPromptRunner

  **增量的输入预处理和模型执行参考**[run_decoder.py](./run_decoder.py)的LlmDecoderRunner

- 加入**分离部署功能**。

  **注意**：下面用不同参数"kv_tensors"和"past_key_values"传递kv cache的tensor主要是区分llama2中的全量和增量执行，不同模型根据实际情况进行调整。

[分离部署接口API参考资料](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/apiref/llmdatadist/llm_python_002.html)

**全量用例执行参考**[run_prompt.py](./run_prompt.py)的main函数

涉及到的分离部署接口示例，实际参考：[run_prompt.py](./run_prompt.py)

```python
prompt_engine = LLMDataDist(LLMRole.PROMPT, cluster_id)
# 分离部署资源初始化
llm_config = LLMConfig()
prompt_engine.init(llm_config.generate_options())

# 申请全量请求kv cache
kv_cache_manager = prompt_engine.kv_cache_manager
kv_cache_desc = CacheDesc(num_tensors=80, shape=[4, 2048, 8 // world_size, 128],
                          data_type=DataType.DT_FLOAT16)
kv_cache_keys = [CacheKey(prompt_cluster_id=0, req_id=0, model_id=0)]
cache = cache_manager.allocate_cache(kv_cache_desc, kv_cache_keys)

# 模型执行，将申请好的kv cache传递给模型，替换原模型中的kv cache tensor
kv_tensor_addrs = cache.per_device_tensor_addrs[0]
kv_tensors = torchair.llm_datadist.create_npu_tensors(cache.cache_desc.shape, torch.float16,                                                                       kv_tensor_addrs)
mid = len(kv_tensors) // 2
k_tensors = kv_tensors[: mid]
v_tensors = kv_tensors[mid:]
kv_cache_tensors = list(zip(k_tensors, v_tensors))
# 此处传递的参数根据不同模型区分全量和增量的入参进行调整
kwargs["kv_tensors"] = kv_cache_tensors
outputs = runner.execute_model(input_tensors, **kwargs)
```

**增量用例执行参考**[run_decoder.py](./run_decoder.py)的main函数

涉及到的分离部署接口示例，实际参考：[run_decoder.py](./run_decoder.py)

```python
decoder_engine = LLMDataDist(LLMRole.DECODER, cluster_id)
# 分离部署资源初始化
llm_config = LLMConfig()
decoder_engine.init(llm_config.generate_options())
# 动态集群链路管理
decoder_engine.link_clusters(clusters_info)

# 申请增量模型kv cache
kv_cache_desc = CacheDesc(num_tensors=80, shape=[4, 2048, 8 // world_size, 128],
                          data_type=DataType.DT_FLOAT16)
cache = cache_manager.allocate_cache(kv_cache_desc)
prompt_cache_key = CacheKey(prompt_cluster_id=0, req_id=0, model_id=0)
# 从全量集群拉取kv cache到指定batch index位置
cache_manager.pull_cache(prompt_cache_key, cache, 0)

# 模型执行，将申请好的kv cache传递给模型，替换原模型中的kv cache tensor
kv_tensor_addrs = cache.per_device_tensor_addrs[0]
kv_tensors = torchair.llm_datadist.create_npu_tensors(cache.cache_desc.shape, torch.float16,                                                                       kv_tensor_addrs)
mid = len(kv_tensors) // 2
k_tensors = kv_tensors[: mid]
v_tensors = kv_tensors[mid:]
kv_cache_tensors = list(zip(k_tensors, v_tensors))
# 此处传递的参数根据不同模型区分全量和增量的入参进行调整
kwargs["past_key_values"] = kv_cache_tensors
outputs = runner.execute_model(input_tensors, **kwargs)
```
