# 固定权重类输入地址功能（aclgraph）

## 功能简介

>**须知：** reduce-overhead模式下的功能为试验特性，后续版本可能存在变更，暂不支持应用于商用产品中。

推理场景下，[Parameter类型](https://pytorch.org/docs/2.1/generated/torch.nn.parameter.Parameter.html)（权重类）的图输入内存地址通常保持不变。可以开启本功能缩短图下发时间，提升下发性能。

对于PyTorch v2.6.0及以上版本，通过[torch.\_dynamo.mark\_static\_address](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_dynamo/decorators.py#L538)接口标记的内存地址不变的图输入Tensor（如LLM模型的kv\_cache）也可以开启本功能。

该功能适用于ChatGPT、LLaMA等开源大模型，请根据自身实际情况开启。

## 使用约束

-   本功能支持如下产品：
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
-   本功能仅支持reduce-overhead模式。

## 使用方法

该功能通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数介绍参见下表。

```python
import torch_npu, torchair
config = torchair.CompilerConfig()
# 设置图执行模式
config.mode = "reduce-overhead"
# 固定权重类输入地址开关
config.experimental_config.frozen_parameter = True
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

**表 1**  参数说明


| 参数名 | 说明 |
| --- | --- |
| frozen_parameter | 图执行时是否固定权重类输入地址。<br>- False（默认值）：不固定权重类输入地址。<br>- True：固定权重类输入地址。 |

## 特殊场景

> **说明：** 
>PyTorch的to算子转换时会丢失Parameter类型，因此需要先将CPU Tensor转换为NPU Tensor，再通过torch.nn.Parameter\(Tensor\)等方式，将普通Tensor转换为Parameter类型的Tensor。

PyTorch的to算子转换示例如下：

```python
import torch
import torch_npu
import torchair

config = torchair.CompilerConfig()
config.experimental_config.frozen_parameter = True
npu_backend = torchair.get_npu_backend(compiler_config=config)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, z):
        return torch.add(x, y*z)
 
model = Model()
# 正确转换方式：先将Tensor转换为NPU Tensor，再转换为Parameter类型，转换后in1是Parameter类型
in1 = torch.nn.Parameter(torch.randn(4, 1).float().npu())
# 错误转换方式：先转换为Parameter类型，再将Tensor转换为NPU Tensor，转换后的in1不是Parameter类型
# in1 = torch.nn.Parameter(torch.randn(4, 1).float()).npu()
in2 = torch.randn(4, 4).float().npu()
in3 = torch.randn(4, 4).int().npu()
model = torch.compile(model, backend=npu_backend, dynamic=True)
graph_result = model(in1, in2, in3)
```
