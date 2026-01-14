# View类算子优化功能

## 功能简介

以图模式方式调用算子时，如果存在多个View类算子，会带来冗余计算，增加计算耗时。默认情况下，TorchAir内部会开启View类算子优化功能，以提升算子执行性能。如果用户需要进行算子调优，尤其是精度比对，建议关闭本功能避免影响调优效果。

> **说明：** 
>本功能当前仅针对如下ATen IR进行优化：
>-   [torch.permute](https://pytorch.org/docs/2.1/generated/torch.permute.html#torch-permute)
>-   [torch.t](https://pytorch.org/docs/2.1/generated/torch.t.html#torch-t)
>-   [torch.transpose](https://pytorch.org/docs/2.1/generated/torch.transpose.html#torch.transpose)
>-   [torch.Tensor.view](https://pytorch.org/docs/2.1/generated/torch.Tensor.view.html#torch.Tensor.view)
>-   [torch.reshape](https://pytorch.org/docs/2.1/generated/torch.reshape.html#torch-reshape)

## 使用约束

本功能仅支持max-autotune模式。

## 使用方法

该功能通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数介绍参见下表。

```python
import torch_npu
import torchair
config = torchair.CompilerConfig()
# View类算子优化配置
config.experimental_config.enable_view_optimize = False
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

**表 1**  参数说明


| 参数名 | 说明 |
| --- | --- |
| enable_view_optimize | 图模式调用View算子时是否开启计算优化。<br>- False：关闭优化。<br>- True（默认值）：开启优化。 |

