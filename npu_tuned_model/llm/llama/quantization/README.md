# Llama2 A8W8C8量化

本模块主要是llama2模型在npu上的完成适配迁移的基础上，介绍如何进行A8W8+KVCach int8(简称A8W8C8)的适配，其中transformers==4.31.0版本。

# 下载模型权重及模型结构
llama2-7B https://huggingface.co/FlagAlpha/Atom-7B/tree/v1.0.0

# 图模式适配

[模型迁移指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/devguide/moddevg/torchair/torchair_01_0001.html)

此适配点主要是加入走pytorch图模式分支。

```python
# transformers/generation/utils.py的greedy_search函数while True前添加
import os
import time
import logging

exe_mode = os.getenv("EXE_MODE", "dynamo")
dynamic_compile = True # 因为当模型结构使能了actual_seq_length

if exe_mode == "dynamo":
    logging.info("Start to run model in dynamo mode, dynamic=%s, fullgraph=%s, backend=npu" % (dynamic_compile, True))
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
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

# A8W8C8适配
**注意**：模型适配A8W8C8之前，请先完成eager模式或者dynamo图模式 非量化场景的改造，然后再进行下面A8W8C8文件的适配。

该目录下的模型结构文件是基于非量化常见的模型结构进行改造适配的，被修改的原函数都加了'__'前缀，可用于对比修改后的函数变化。下面列出了优化项主要的改动，不同的大模型适配完主要改动后，在根据模型实际代码逻辑进行调试。

## 重定义Linear的A8W8类
由于CANN提供了torch api接口npu_quant_matmul，可支持matmul的输入x为int8，且weight也为int8数据类型，我们可以根据需要重新定义一个A8W8的Module类，假设类名为NpuA8W8Linear，并保存为 npu_quantize.py 文件。
```python
import os
import sys
import torch
from torch import nn
import torch_npu


class NpuA8W8Linear(torch.nn.Module):
    __constants__ = ["in_feature", "out_feature"]
    in_feature: int
    out_feature: int
    def __init__(self, in_feature: int, out_feature: int, device="npu",
                output_dtype=torch.float16) -> None:
        super().__init__()
        self.output_dtype = output_dtype
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        ## A8W8 weight Perchannel symmetry, x pertensor asymmetry
        self.register_buffer("bias", torch.ones(self.out_feature, dtype=torch.int32, device="npu"))
        self.register_buffer("weight", torch.ones((self.out_feature, self.in_feature), dtype=torch.int8, device="npu"))
        self.register_buffer("deq_scale", torch.ones(self.out_feature, dtype=torch.int64, device="npu"))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.out_feature, )
        x = x.view(-1, x.size(-1))

        y = torch_npu.npu_quant_matmul( x,
                                        self.weight.transpose(0,1),
                                        self.deq_scale,
                                        offset=None,
                                        bias=self.bias,
                                        output_dtype=self.output_dtype)

        return y.view(size_out)
```

## 模型主体结构适配

### Attention中QKVO的A8W8适配
**改造原因**：将原来非量化的 Linear 接口换成 NpuA8W8Linear 接口。
```python
''' LlamaAttention类中 _init_ 初始化 替换部分，替换前如下
self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
self.qkv = nn.Linear(self.hidden_size,
                      self.num_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim, bias=False)
'''

### 替换后如下
self.o_proj = NpuA8W8Linear(self.q_hidden_size, self.hidden_size)
self.qkv = NpuA8W8Linear(self.hidden_size,  self.q_hidden_size + 2 * self.kv_hidden_size)

## qkvo activation asymmetric quantization parameters
self.register_buffer("scale_qkv", torch.ones(self.hidden_size, dtype=torch.float32, device="npu"))
self.register_buffer("offset_qkv", torch.ones(self.hidden_size, dtype=torch.int32, device="npu"))
self.register_buffer("scale_o", torch.ones(self.q_hidden_size, dtype=torch.float32, device="npu"))
self.register_buffer("offset_o", torch.ones(self.q_hidden_size, dtype=torch.int32, device="npu"))

## kvcache int8 asymmetric quantization parameters
self.register_buffer("kcache_scale", torch.ones(self.kv_hidden_size, dtype=torch.float32, device="npu"))
self.register_buffer("kcache_offset", torch.ones(self.kv_hidden_size, dtype=torch.int32, device="npu"))
self.register_buffer("vcache_scale", torch.ones(self.kv_hidden_size, dtype=torch.float32, device="npu"))
self.register_buffer("vcache_offset", torch.ones(self.kv_hidden_size, dtype=torch.int32, device="npu"))
self.register_buffer("ifa_antiquant_scale", torch.ones((2, self.kv_hidden_size), dtype=torch.float16, device="npu"))
self.register_buffer("ifa_antiquant_offset", torch.ones((2, self.kv_hidden_size), dtype=torch.float16, device="npu"))
```

```python
## LlamaAttention类中 forward 在qkv计算前新增一行，完成输入从FP16->int8的量化
hidden_states = torch_npu.npu_quantize(hidden_states, self.scale_qkv, self.offset_qkv, torch.qint8, axis=-1)
## forward中 在o_proj计算前新增一行，完成输入从FP16->int8的量化
attn_output = torch_npu.npu_quantize(attn_output, self.scale_o, self.offset_o, torch.qint8, axis=-1)
```

### MLP中A8W8适配
**改造原因**：将原来非量化的 Linear 接口换成 NpuA8W8Linear 接口。
```python
''' LlamaMLP类中 _init_ 初始化 替换部分，替换前如下
self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
'''

### 替换后如下
self.gate_proj = NpuA8W8Linear(self.hidden_size, self.intermediate_size) # 替换
self.up_proj = NpuA8W8Linear(self.hidden_size, self.intermediate_size)   # 替换
self.down_proj = NpuA8W8Linear(self.intermediate_size, self.hidden_size) # 替换
...
self.register_buffer("scale_up", torch.ones(self.hidden_size, dtype=torch.float32, device="npu"))  # 新增
self.register_buffer("offset_up", torch.ones(self.hidden_size, dtype=torch.int32, device="npu"))   # 新增
self.register_buffer("scale_down", torch.ones(self.intermediate_size, dtype=torch.float32, device="npu")) # 新增
self.register_buffer("offset_down", torch.ones(self.intermediate_size, dtype=torch.int32, device="npu"))  # 新增
```

```python
''' LlamaMLP类中 forward 替换部分，替换前如下
down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
'''

### 替换后如下
x = torch_npu.npu_quantize(x, self.scale_up, self.offset_up, torch.qint8, axis=-1)
x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
x = torch_npu.npu_quantize(x, self.scale_down, self.offset_down, torch.qint8, axis=-1)
return self.down_proj(x)
```

### KVCache int8的适配
**改造原因**：由于KVCache做了量化，它采用了int8数据类型存储。相比FP16存储，内存上可以省一半。同时，由于增量图在进行FlashAttention时，取了KVCache的值，需要给 npu_incre_flash_attention 传反量化参数scale和offset。
```python
### 在 apply_rotary_pos_emb 实现后新增如下几行，完成Key/Value从FP16->INT8的量化，从而更新到KVCache中的数据为INT8。
### 本示例中KVCache采用的是 perchannel asymmetric 量化。
rshp_key = key_states.reshape(bsz, q_len, self.kv_hidden_size)
quant_key = torch_npu.npu_quantize(rshp_key, self.kcache_scale, self.kcache_offset, torch.qint8, axis=-1)
quant_key = quant_key.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
rsh_value = value_states.reshape(bsz, q_len, self.kv_hidden_size)
quant_value = torch_npu.npu_quantize(rsh_value, self.vcache_scale, self.vcache_offset, torch.qint8, axis=-1)
quant_value = quant_value.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
```

```python
''' npu_incre_flash_attention 接口替换前如下
attn_output = torch_npu.npu_incre_flash_attention(query_states, key_states1.contiguous(),
                                                  value_states1.contiguous(), num_heads=self.num_heads,
                                                  input_layout="BSND",
                                                  scale_value=self.scale_value,
                                                  atten_mask=attention_mask,
                                                  actual_seq_lengths=actual_seq_len,
                                                  kv_padding_size=kv_padding_size,
                                                  num_key_value_heads=self.num_key_value_heads)
'''

### 替换后如下
attn_output = torch_npu.npu_incre_flash_attention(query_states, key_states1.contiguous(),
                                                  value_states1.contiguous(), num_heads=self.num_heads,
                                                  input_layout="BSND",
                                                  scale_value=self.scale_value,
                                                  antiquant_scale=self.ifa_antiquant_scale,   # 新增
                                                  antiquant_offset=self.ifa_antiquant_offset, # 新增
                                                  atten_mask=attention_mask,
                                                  actual_seq_lengths=actual_seq_len,
                                                  kv_padding_size=kv_padding_size,
                                                  num_key_value_heads=self.num_key_value_heads)
```

# 性能数据

## llama2-7B

执行llama2-7b，由于没有真实的量化权重及参数，加载的模型路径我们依旧使用FP16的原始权重，不存在在参数使用随机初始化值，输入padding到1024长度，输出max_new_tokens是1024的性能数据如下：

**在800I A2的机器上，host是arm，4batch size性能&内存数据**

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">TP</th>
    <th class="tg-0pky" rowspan="2">batch size</th>
    <th class="tg-0pky" rowspan="2">量化模式</th>
    <th class="tg-0pky" colspan="2">图模式</th>
    <th class="tg-0pky" rowspan="2">内存占用(MB)</th>
  </tr>
  <tr>
    <th class="tg-0pky">全量(ms)</th>
    <th class="tg-0pky">增量(ms)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">1</td>
    <td class="tg-0pky">4</td>
    <td class="tg-0pky">/</td>
    <td class="tg-0pky">309</td>
    <td class="tg-0pky">23.9</td>
    <td class="tg-0pky">21688</td>
  </tr>
  <tr>
    <td class="tg-0pky">1</td>
    <td class="tg-0pky">4</td>
    <td class="tg-0pky">A8W8C8</td>
    <td class="tg-0pky">203</td>
    <td class="tg-0pky">17.8</td>
    <td class="tg-0pky">11758</td>
  </tr>
</tbody>
</table>

# 性能测试

在benchmark目录下的npu目录提供了llama执行样例参考。上述性能数据在arm host + 800I A2环境执行进行统计 

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
# torch dynamo图模式 A8W8C8
python3 benchmark/npu/benchmark_llama.py --model_path=xxx/llama2-7b/fp16_model_dir/ --execute_mode=dynamo --quant_mode="a8w8c8"
# torch eager 单算子模式 A8W8C8
python3 benchmark/npu/benchmark_llama.py --model_path=xxx/llama2-7b/fp16_model_dir/ --execute_mode=eager --quant_mode="a8w8c8"
```

**性能数据打点位置**

参考图模式适配模型执行部分修改。全量耗时统计的是模型第一个step数据，增量耗时统计后续step数据的均值。