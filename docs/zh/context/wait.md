# wait

## 支持的型号

<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>

<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 功能说明

用于在多流间控制时序关系，torchair.ops.wait表示当前流需要在传入的tensor对应的节点执行结束后，再继续执行。

## 函数原型

```python
wait(tensors: List[torch.Tensor])
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| tensors | 输入 | 当前流需要等待的tensor，可以传入多个tensor。 | 是 |

## 返回值说明

无

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
# config.mode = "reduce-overhead" 
npu_backend = torchair.get_npu_backend(compiler_config=config)
func = torch.compile(demo, backend=npu_backend, dynamic=False, fullgraph=True)
input1 = torch.ones(2, 2).npu()
input2 = torch.ones(2, 2).npu()
func(input1, input2)
```

