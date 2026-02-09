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
device_npu = 'npu:0'
mm_input = torch.randn(3200, 3200, device=device_npu)

def func(input):
    global mm_input
    A = torch.zeros([100, 100], device=device_npu) # A在主流上申请内存, 全0
    with torchair.scope.npu_stream_switch('second_stream', 3):
        for _ in range(2000):  # 延长secend stream执行时间，使得A.add(1)晚于主流C.add_(2)计算
            mm_input = mm_input @ mm_input
        B = A.add(1)
        # A在secend_stream参与计算，同时主流对A所在内存进行释放，此时，需要插入record_stream延长A所在内存的生命周期，
        # 避免被提前释放, 导致出现A在secend stream计算时数据错误改写的问题
        torchair.ops.npu_record_tagged_stream(A, 'second_stream') # 延长A内存生命周期, 不加这一行，B的输出结果可能不是1, 加了这一行，B的输出结果一定是1
    del A # 在主流上释放A内存，如果在second_stream流上没有插入record_stream，则可能导致A内存被提前释放，
    # 而正好C又恰好申请到了A相同的内存地址，并改写了数据，导致B的结果错误
    C = torch.ones([100, 100], device=device_npu) # A在主流上申请内存, 全1
    C.add_(100)
    return B, C, mm_input
config = CompilerConfig()
config.mode = "reduce-overhead"
npu_backend = torchair.get_npu_backend(compiler_config=config)

# 调用compile编译
func = torch.compile(func, backend=npu_backend, dynamic=False, fullgraph=True)
in1 = torch.ones([100, 100], device=device_npu)

result = func(in1)
print(result[0])
print(result[1])
```
