# npu\_record\_tagged\_stream

## 支持的型号

<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>

<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 功能说明

与PyTorch原生[torch.Tensor.record\_stream接口](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)功能类似，用于确保张量在特定NPU流完成之前不会被释放，该方法对张量的内存管理和异步操作非常重要。详细功能介绍参见[图内多流表达功能（aclgraph）](图内多流表达功能（aclgraph）.md)。

当在多stream之间共享张量时，本接口实现如下功能：

-   记录使用该张量的stream，延长张量的生命周期
-   防止张量在使用它的流完成工作前被释放
-   确保正确的内存管理

具体来说，当某Tensor在stream\_i上使用，而其它stream上释放该Tensor时，系统不会真正释放内存回内存池，而是会插入一个event，只有等待event执行完成后，该Tensor持有的内存才会被真正释放。

注意：原生接口无法入FX图，本接口支持入图。

## 函数原型

```python
npu_record_tagged_stream(input: torch.Tensor, tagged_stream: str)
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| input | 输入 | 待延长生命周期的Tensor，一般是某个stream申请，另外一个stream使用。 | 是 |
| tagged_stream | 输入 | 该Tensor被某个tag stream使用，如果是默认stream，请设置为tagged_stream="default"。 | 是 |

## 返回值说明

无

## 约束说明

-   本接口只在reduce-overhead模式下生效，其他模式不建议使用。
-   其他约束与torch.Tensor.record\_stream保持一致，此处不再赘述。

## 调用示例

```python
import torch, os
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger

def func(input):
    mm_input = torch.randn(3200, 32000)
    with torchair.scope.npu_stream_switch('1'):
        # 延长second stream执行时间，使得B = input + input晚于主流C.add_(2)计算
        for _ in range(100):                
            out = mm_input * mm_input
        B = input + input
        # 调用npu_record_tagged_stream，表明Tensor B在stream'1'上使用，延长Tensor B对应内存的生命周期
        torchair.ops.npu_record_tagged_stream(B, '1')
    del B
    C = torch.ones([100, 100], device="npu")
    C.add_(2)
    return C
config = CompilerConfig()
config.mode = "reduce-overhead"
npu_backend = torchair.get_npu_backend(compiler_config=config)

# 调用compile编译
func = torch.compile(func, backend=npu_backend, dynamic=False, fullgraph=True)
in1 = torch.ones([100, 100], device="npu")
result = func(in1)
print(f"Result:\n{result}\n")
```
