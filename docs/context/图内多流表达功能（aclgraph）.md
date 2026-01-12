# 图内多流表达功能（aclgraph）

## 功能简介

> **须知：** reduce-overhead模式下的功能为试验特性，后续版本可能存在变更，暂不支持应用于商用产品中。

大模型推理场景下，对于一些可并行的场景，可以划分多个stream提升执行效率。通过在脚本中指定每个算子的执行stream，将原本需要串行的多个算子分发到不同stream做并行计算，多个stream上的计算形成overlap，从而降低整体计算耗时。

对于并行来说，包含如下两种：

-   计算与计算并行：一般是基于数据依赖关系，分析出可以并行的多条计算分支，指定stream并行。
-   计算与通信并行：一般是针对没有数据依赖的通信操作，提前使用通信资源执行通信任务。

本功能主要处理**aclgraph间资源并发（reduce-overhead模式）**，尤其针对Cube计算资源未完全使用的场景。若Cube计算资源已完全使用，不建议开启本功能，可能会造成额外的调度，从而导致原计算性能劣化。

## 使用约束

-   本功能支持如下产品：
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
-   本功能仅支持reduce-overhead模式。
-   本功能不支持与[FX Pass配置功能](FX-Pass配置功能.md)中reinplace相关Pass同时开启。换言之，如果图内存在多流并行，此类Pass开启了也不会生效。

## 使用方法

1.  用户自行分析模型脚本中可进行并行计算的算子。
2.  开启图内多流表达。

    使用如下with语句块（[npu\_stream\_switch](npu_stream_switch.md)），语句块内下发的算子切换至stream\_tag流，语句块外的算子使用默认stream计算。

    ```python
    with torchair.scope.npu_stream_switch(stream_tag: str, stream_priority: int = 0)
    ```

    -   stream\_tag：表示需要切换到的流的标签，相同的标签代表相同的流，由用户控制。
    -   stream\_priority：表示切换到stream\_tag流的优先级，即Runtime运行时在并发时优先给高优先级的流分配核资源，当前版本使用默认值0即可。

3.  （可选）控制并行计算的时序。

    通过[npu\_create\_tagged\_event](npu_create_tagged_event.md)、[npu\_tagged\_event\_record](npu_tagged_event_record.md)、[npu\_tagged\_event\_wait](npu_tagged_event_wait.md)系列接口实现时序控制。接口功能与torch.npu.Event使用对等，表明wait需要等待record执行完后才会执行。若用户脚本中多流场景下的算子存在输入输出依赖，需通过上述系列接口显式控制时序，避免因并行乱序执行导致计算逻辑异常或数值精度问题。

4.  （可选）延长内存释放时机。

    Eager模式场景下，脚本中如果涉及多stream内存复用，一般会调用PyTorch的tensor.record\_stream接口延迟内存释放。由于该接口无法入FX图，因此TorchAir提供了reduce-overhead图模式下对等的API  [npu\_record\_tagged\_stream](npu_record_tagged_stream.md)。

## 使用示例

```python
import torch, os
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger

# 定义模型model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建两个tag="66"，tag="77"的event
        self.tagged_event1 = tng.ops.npu_create_tagged_event(tag="66")
        self.tagged_event2 = tng.ops.npu_create_tagged_event(tag="77")
    def forward(self, in1, in2, in3, in4):
        add_result = torch.add(in1, in2)
        # 插入一个event_record用于同步，对于self.tagged_event1.wait后的任务需要等record执行完毕才能执行
        tng.ops.npu_tagged_event_record(self.tagged_event1)
        with tng.scope.npu_stream_switch('1'):
            # torch.mm算子(mm_result)等待torch.add算子(add_result)执行完再执行
            tng.ops.npu_tagged_event_wait(self.tagged_event1)
            mm_result = torch.mm(in3, in4)
            # 插入一个event_record用于同步，对于self.tagged_event2.wait后的任务需要等record执行完毕才能执行
            tng.ops.npu_tagged_event_record(self.tagged_event2)
            B = in3 + in4
            # 调用npu_record_tagged_stream，表明Tensor B在stream'1'上使用，延长Tensor B对应内存的生命周期
            tng.ops.npu_record_tagged_stream(B, '1')
        mm1 = torch.mm(in3, in4)
        del B
        C = torch.ones(1000, 1000, dtype = torch.float16, device="npu")
        C.add_(2)
        with tng.scope.npu_stream_switch('2'):
            # torch.add算子(add2)等待torch.mm算子(mm_result)执行完再执行
            tng.ops.npu_tagged_event_wait(self.tagged_event2)
            add2 = torch.add(in3, in4)
        return add_result, mm_result, mm1, add2, C
model = Model()
config = CompilerConfig()
config.mode = "reduce-overhead"
npu_backend = tng.get_npu_backend(compiler_config=config)

# 使用torchair的backend去调用compile接口编译模型
model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
in1 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in2 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in3 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in4 = torch.randn(1000, 1000, dtype = torch.float16).npu()
result = model(in1, in2, in3, in4)
print(f"Result:\n{result}\n")
```
