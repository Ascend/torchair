

# 分离部署迁移指南

本章节主要介绍如何将一个pytorch的大模型脚本迁移为可以全量和增量分离部署的脚本

# 迁移步骤

- 准备一个**能够成功执行**推理的大模型脚本。

- 查看脚本执行推理的代码**是否是调用的集成接口**。什么是集成接口，就是调用一次可以完成整个句子长度推理的接口。transformers提供的generate接口示例：

  ```python
  # 参考runner/llm_runner.py的LlmModelRunner.model_generate函数
  if self.input_padding:
      inputs = self.tokenizer(prompts,
                              return_tensors="pt",  # 返回pytorch tensor
                              truncation=True,
                              padding='max_length',
                              max_length=kwargs.get("input_max_len", 1024))
  else:
      inputs = self.tokenizer(prompts,
                              return_tensors="pt",  # 返回pytorch tensor
                              truncation=True)
  
  kwargs_params = self._generate_params(inputs, kwargs.get("max_new_tokens", 1024))
  start_time = time.time()
  with torch.no_grad():
      generate_ids = self.model.generate(**kwargs_params)
  elapse = time.time() - start_time
  ```

- 如果是集成接口，那么需要做的**调整脚本执行推理的代码，修改为一次调用只会推理一次**。pytorch主要修改为直接调用模型定义的forward函数。一次调用只做一次推理的示例：

  ```python
  # 参考runner/separate_deployment/llm_inference.py的SeparateDeployModelRunner.model_generate函数
  torch.npu.synchronize()
  outputs = self.model(**model_inputs)
  torch.npu.synchronize()
  next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)
  ```

- 调整成一次调用只会推理一次后，需要**保证能够和使用集成接口精度和性能一致**

- 上述做完后，根据模型结构定义的代码如何区分全量和增量执行的，将**执行脚本拆分为全量执行脚本和增量执行脚本**。大模型脚本多以seq_length和kv cache区分全量和增量执行。

  以llama2为例，llama2中通过past_key_values这个输入是否是None，区分了全量和增量。当past_key_values为None时是全量推理，非None时时增量推理。示例代码：

  ```python
  # 参考models/llama/modeling_llama.py的LlamaForCausalLM.prepare_inputs_for_generation函数
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

**全量用例执行参考**[run_prompt.py](./run_prompt.py)的main函数

涉及到的分离部署接口：

```python
engine = LLMEngine(LLMRole.PROMPT, cluster_id)
# 分离部署资源初始化
engine.init(options)
# 自定义模型执行接口，将申请好的kv cache tensor传入适配好的模型
class TorchModelRunner(ModelRunner):
    def __init__(self, model_runner, kv_cache_manager):
        self._model_runner = model_runner
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
# 添加engine执行的模型
llm_model = engine.add_model(model_options, TorchModelRunner(model_runner, engine.kv_cache_manager))
# 创建llm_req
llm_req = LLMReq()
# 模型执行
result = llm_model.predict(llm_reqs[4:], [inputs], **config)
```

**增量用例执行参考**[run_decoder.py](./run_decoder.py)的main函数

涉及到的分离部署接口：

```python
engine = LLMEngine(LLMRole.DECODER, cluster_id)
# 分离部署资源初始化
engine.init(options)
# 自定义模型执行接口，将申请好的kv cache tensor传入适配好的模型
class TorchModelRunner(ModelRunner):
    def __init__(self, model_runner, kv_cache_manager):
        self._model_runner = model_runner
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
# 添加engine执行的模型
llm_model = engine.add_model(model_options, TorchModelRunner(model_runner, engine.kv_cache_manager))
# 创建llm_req
llm_req = LLMReq()
# 从全量拉kv和merge kv到指定batch位置
llm_model.pull_kv(llm_req)
llm_model.merge_kv(llm_req, i)
# 模型执行
result = llm_model.predict(llm_reqs, [inputs], **config)
```

# 性能及功能测试

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

[llm_link_torch包安装](https://gitee.com/cann/air/blob/ge_dev/python/llm_link_torch/README.md)

**设置环境变量**

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/your/torchair/npu_tuned_model/llm/llama
cann_path=/usr/local/Ascend # 昇腾cann包安装目录
source ${cann_path}/latest/bin/setenv.bash
export ASCEND_HOME_PATH=${cann_path}/latest
export RESOURCE_CONFIG_PATH=numa_config.json # numa_config配置参考内部资料
```

**qkv权重融合**

```shell
model_path=xxx/llama2-70b # 下载的权重和模型信息
python3 merge_qkv_weight.py --model_path=${model_path} --tp_size=8 --output_path=xxx/llama-70b_qkv
```

**将替换了mc2融合算子的LinearAllreduce替换deepspeed原生的LinearAllreduce**

将benchmark/deepspeed/mc2_adapter.py的LinearAllreduce整个类拷贝替换原生deepspeed的deepspeed/module_inject/layers.py中的LinearAllreduce类，并且import torch_npu

**注：上述操作分别在全量和增量机器上完成后，**

先拉起全量的执行脚本

```shell
deepspeed --num_gpus=8 benchmark/pd_separate/run_prompt.py --model_path=xxx/llama2-70b_qkv
```

将目录下生成的prompt.pkl文件拷贝到增量执行目录下

```shell
deepspeed --num_gpus=8 benchmark/pd_separate/run_decoder.py --model_path=xxx/llama2-70b_qkv
```

**打点位置**

在llm_inference.py的model_generate函数中start和end