# get\_compiler

## 功能说明

获取能够在NPU上运行的图编译器，可以将获取的图编译器传入自定义的后端中，以实现用户自定义的特性。

## 函数原型

```python
get_compiler(compiler_config: CompilerConfig = None)
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| compiler_config | 输入 | 图编译配置，[CompilerConfig类](CompilerConfig类.md)的实例化，默认情况下采用TorchAir自动生成的配置。 | 否 |

## 返回值说明

返回NpuFxCompiler。

## 约束说明

无

## 调用示例

用户自定义一个可以打印GM的backend，通过torchair.get\_compiler获取NPU compiler。

```python
import os
import torch
from torch._functorch.aot_autograd import aot_module_simplified
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig

class MM(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x = x + y
        return x
def custom_backend(gm: torch.fx.GraphModule, example_inputs):
    compiler_config = CompilerConfig()
    compiler = torchair.get_compiler(compiler_config)
    print(gm)
    return aot_module_simplified(gm, example_inputs, fw_compiler=compiler)
                        
torch.npu.set_device(0)
x = torch.ones([2, 2], dtype=torch.int32).npu()
y = torch.ones([2, 2], dtype=torch.int32).npu()
model = torch.compile(MM().npu(), backend=custom_backend, dynamic=False)
ret = model(x, y)
print(ret)
```

