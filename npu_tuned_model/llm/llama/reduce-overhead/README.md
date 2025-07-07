# Llama2 图模式(reduce-overhead)

本模块主要是llama2模型在npu上的完成适配迁移的基础上，介绍如何进行reduce-overhead模式的适配，其中transformers==4.31.0版本。

# 下载模型权重及模型结构
llama2-7B https://huggingface.co/FlagAlpha/Atom-7B/tree/v1.0.0

# 图模式(reduce-overhead)适配

[模型迁移指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/devguide/moddevg/torchair/torchair_01_0001.html)

**注意**：模型适配reduce-overhead之前，请先完成eager模式的适配，并确保在npu上运行正常。

此适配点主要是让pytorch图模式走reduce-overhead模式进行执行。

# transformers/generation/utils.py的greedy_search函数while True前添加
```python
import os
import time
import logging

exe_mode = os.getenv("EXE_MODE", "dynamo")
graph_mode = os.getenv("GRAPH_MODE", "max-autotune")
dynamic_compile = True # 因为当模型结构使能了actual_seq_length

if exe_mode == "dynamo":
    logging.info("Start to run model in dynamo mode, dynamic=%s, fullgraph=%s, backend=npu" % (dynamic_compile, True))
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    if graph_mode == "reduce-overhead":
        config.mode = "reduce-overhead" # 修改为reduce-overhead模式
    config.experimental_config.frozen_parameter = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    self = torch.compile(self, dynamic=dynamic_compile, fullgraph=True, backend=npu_backend)
else:
    logging.info("Start to run model in eager(HOST API) mode")

while True:
    if synced_gpus:
        ...

# 在模型执行前后添加torch.npu.synchronize()，主要是做性能统计，打点位置参考如下
torch.npu.synchronize()
start = time.time()
outputs = self(
    **model_inputs,
    return_dict=True,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
)
torch.npu.synchronize()
end = time.time()
cost = end - start # 每个step的耗时统计
```

- torchair提供了NPU的图构造/图编译/图执行能力。相关能力全部集成到NPU图的后端，在使用torch.compile接口时，指定NPU图后端来使能。同时提供了开关和config控制图的编译和执行流程。
- 在使用NPU图后端的时候，torchair提供了静态图和动态图两种图执行的能力。根据dynamic参数决定是否走动态图。
- 使用reduce-overhead模式时，如果torch为2.5以下的版本，由于原生缺陷可能在使能reinplace_inplaceable_ops_pass时抛出异常，有2种解决方法：1.pass不关闭：升级torch>2.5.0 2.pass关闭，不升级torch，需配置：config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass=True

# 性能数据

## llama2-7B

执行llama2-7b，加载模型时的tensor类型float16，输入padding到1024长度，输出max_new_tokens是1024的性能数据如下：

**在800I A2的机器上，host是arm，4batch size性能&内存数据**

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">模式</th>
    <th class="tg-0pky" rowspan="2">batch size</th>
    <th class="tg-0pky" colspan="2">性能耗时</th>
    <th class="tg-0pky" rowspan="2">内存占用(MB)</th>
  </tr>
  <tr>
    <th class="tg-0pky">全量(ms)</th>
    <th class="tg-0pky">增量(ms)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">单算子</td>
    <td class="tg-0pky">4</td>
    <td class="tg-0pky">207</td>
    <td class="tg-0pky">27.8</td>
    <td class="tg-0pky">21625</td>
  </tr>
  <tr>
    <td class="tg-0pky">图模式(max-autotune)</td>
    <td class="tg-0pky">4</td>
    <td class="tg-0pky">208</td>
    <td class="tg-0pky">18.6</td>
    <td class="tg-0pky">21633</td>
  </tr>
    <tr>
    <td class="tg-0pky">图模式(reduce-overhead)</td>
    <td class="tg-0pky">4</td>
    <td class="tg-0pky">229</td>
    <td class="tg-0pky">27.9</td>
    <td class="tg-0pky">20943</td>
  </tr>
</tbody>
</table>

# 精度数据

得到的精度结果（仅参考）如下：

- 问题1：用一句话描述地球为什么是独一无二的。
- 回答：地球是太阳系唯一有生命存在的星球

- 问题2：给出一段对话，使用合适的语气和回答方式继续对话。对话：A：你今天看起来很高兴，发生了什么好事？B：是的，我刚刚得到一份来自梅西银行的工作通知书。A：哇，恭喜你！你打算什么时候开始工作？B：下个月开始，所以我现在正为这份工作做准备。"
- 回答：A：你打算做什么？B:我打算先学习一些关于银行的基本知识，然后开始我的工作。A：听起来不错。祝你工作顺利！B：谢谢你，祝你一切顺利！

- 问题3：What's your name?
- 回答：My name is Li Ming.李明

- 问题4：Why is the sky blue?
- 回答：The sky is blue because of the scattering of light by the atmosphere.

# 性能精度测试

在benchmark目录下的npu目录提供了llama执行样例参考。上述性能和精度数据在arm host + 800I A2环境执行进行统计 

**基于搭建的conda环境，安装对应的transformers版本**

```shell
# llama2
pip3 install transformers==4.31.0
```

**根据图模式适配方式替换transformers中的utils.py**

```shell
transformers_path=$(pip3 show transformers|grep Location|awk '{print $2}')
# 根据图模式适配方式修改${transformers_path}/transformers/generation/utils.py中的函数
```

**设置环境变量**

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/your/torchair/npu_tuned_model/llm/llama
cann_path=/usr/local/Ascend/ascend-toolkit # 以root用户安装的toolkit包目录
source ${cann_path}/setenv.bash
```

**拉起脚本&执行**

```shell
# torch dynamo图模式(max-autotue)
python3 reduce-overhead/benchmark_llama.py --model_path=xxx/llama2-7b/fp16_model_dir/ --execute_mode=dynamo --graph_mode=max-autotune
# torch dynamo图模式(reduce-overhead)
python3 reduce-overhead/benchmark_llama.py --model_path=xxx/llama2-7b/fp16_model_dir/ --execute_mode=dynamo --graph_mode=reduce-overhead
# torch eager 单算子模式 
python3 reduce-overhead/benchmark_llama.py --model_path=xxx/llama2-7b/fp16_model_dir/ --execute_mode=eager
```

**性能数据打点位置**

参考图模式适配模型执行部分修改。全量耗时统计的是模型第一个step数据，增量耗时统计后续step数据的均值。