# record

## 功能说明

本接口实现了在当前Stream上显示下发一个任务，其返回值可以被torchair.ops.wait等待。在依赖的节点没有输出时，可以使用record。（即使依赖的节点有输出，仍然可以使用torchair.ops.record让FX图看起来更清晰。）

## 函数原型

```python
record()
```

## 参数说明

无

## 返回值说明

返回一个tensor，可以被torchair.ops.wait等待。

## 约束说明

无

## 调用示例

```python
import torch
import torch_npu
import torchair
from torchair import CompilerConfig

def demo(x, y):
    with torchair.scope.npu_stream_switch('1'):
        mm = torch.mm(x, x)        
        abs = torch.abs(mm)
        record = torchair.ops.record()
    add = torch.add(abs, 1)
    torchair.ops.wait([record])
    sub = torch.sub(x, mm)
    return add, sub

config = CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
func = torch.compile(demo, backend=npu_backend, dynamic=False, fullgraph=True)
input1 = torch.ones(2, 2).npu()
input2 = torch.ones(2, 2).npu()
func(input1, input2)
```
