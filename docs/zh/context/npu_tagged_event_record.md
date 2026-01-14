# npu\_tagged\_event\_record

## 支持的型号

<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>

<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 功能说明

与torch.npu.Event.record方法类似（torch.cuda.Event.record的NPU形式），用于记录当前流中的事件。

当算子下发时，系统会获取用户设置的所属流（stream）标签信息，并在该流上下发一个record任务。该任务用于记录NPU流上的一个事件，该事件可用来同步流的执行，与[npu\_tagged\_event\_wait](npu_tagged_event_wait.md)配套使用。详细功能介绍参见[图内多流表达功能（aclgraph）](图内多流表达功能（aclgraph）.md)。

## 函数原型

```python
npu_tagged_event_record(event)
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| event | 输入 | 通过[npu_create_tagged_event](npu_create_tagged_event.md)接口创建出来的event。 | 是 |

## 返回值说明

无

## 约束说明

-   本接口只在reduce-overhead模式下生效，其他模式不建议使用。
-   其他约束与torch.npu.Event.record保持一致，此处不再赘述。

## 调用示例

```python
import torch, os
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger

# 创建一个tag标识为"66"的event对象
GLOBAL_EVENT = torchair.ops.npu_create_tagged_event(tag="66")

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, in1, in2, in3, in4):
        global GLOBAL_EVENT
        add_result = torch.add(in1, in2)
        # 插入一个event_record用于device上不同流之间的同步，对于GLOBAL_EVENT的wait后的任务需要等record执行完毕才能执行
        torchair.ops.npu_tagged_event_record(GLOBAL_EVENT)
        with torchair.scope.npu_stream_switch('1'):
            # torch.mm算子(mm_result)等待torch.add算子(add_result)执行完再执行
            torchair.ops.npu_tagged_event_wait(GLOBAL_EVENT)
            mm_result = torch.mm(in3, in4)
        mm1 = torch.mm(in3, in4)
        add2 = torch.add(in3, in4)
        return add_result, mm_result, mm1, add2
model = Model()
config = CompilerConfig()
config.mode = "reduce-overhead"
npu_backend = torchair.get_npu_backend(compiler_config=config)

# 调用compile编译
model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
in1 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in2 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in3 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in4 = torch.randn(1000, 1000, dtype = torch.float16).npu()
result = model(in1, in2, in3, in4)
print(f"Result:\n{result}\n")
```

