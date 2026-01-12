# npu\_create\_tagged\_event

## 支持的型号

<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>

<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 功能说明

根据tag创建一个唯一的事件对象torch.npu.Event（torch.cuda.Event的NPU形式，参见《PyTorch 原生API支持度》中的“torch.cuda”），用于协调NPU上不同stream之间的同步操作。

只有通过该API创建的event才能在reduce-overhead模式下工作，同一个进程内如果tag相同，调用两次会报错。详细功能介绍参见[图内多流表达功能（aclgraph）](图内多流表达功能（aclgraph）.md)。

## 函数原型

```python
npu_create_tagged_event(tag: str)
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| tag | 输入 | 创建唯一event的标记信息，str类型。 | 是 |

## 返回值说明

返回一个torch.npu.Event类型，该类型在reduce-overhead模式下可以入图使用。

## 约束说明

-   本接口只在reduce-overhead模式下生效，其他模式不建议使用。
-   其他约束与torch.cuda.Event保持一致，此处不再赘述。

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

