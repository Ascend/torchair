# get\_npu\_backend

## 功能说明

获取能够在NPU上运行的图编译后端npu\_backend，可作为backend参数传入torch.compile。

## 函数原型

```python
get_npu_backend(*, compiler_config: CompilerConfig = None, custom_decompositions: Dict = {}) -> npu_backend
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| compiler_config | 输入 | 图编译配置，[CompilerConfig类](CompilerConfig类.md)的实例化，默认情况下采用TorchAir自动生成的配置。 | 否 |
| custom_decompositions | 输入 | 指定模型运行时用到的decompositions（将较大算子操作分解为较简单或核心算子），字典类型。 | 否 |
| * | 输入 | 预留参数，用于后续功能扩展。 | 否 |

## 返回值说明

返回编译后端npu\_backend。

## 约束说明

无

## 调用示例

```python
import torch
import torch_npu
import torchair

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.p1 = torch.nn.Parameter(torch.randn(2, 4))
        self.p2 = torch.nn.Parameter(torch.randn(2, 4))

    def forward(self, x, y):
        x = x + y + self.p1 + self.p2
        return x

model = Model().npu()
config = torchair.CompilerConfig()
# 获取NPU提供的默认backend
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

