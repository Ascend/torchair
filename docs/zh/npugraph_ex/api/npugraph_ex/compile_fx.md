# compile\_fx

## 功能说明

获取能够在NPU上运行的图编译器，可以将获取的图编译器传入自定义的后端中，以实现用户自定义的特性。

## 函数原型

```python
compile_fx(options: dict = None)
```

## 参数说明

|**参数**|**输入/输出**|**说明**|
|--|--|--|
|options|输入|模型编译的功能配置项。|


## 返回值说明

返回NpuFxCompiler。

## 约束说明

使用compile_fx接口构建自定义后端时，不支持以下基于npugraph_ex后端实现的功能：

- [图编译Debug信息保存功能](../../dfx/debug_save.md)
- [多流表达功能](../../advanced/multi_stream.md)
- [AI-Core和Vector-Core限核功能](../../advanced/limit_cores.md)
- [FX图优化Pass配置功能](../../basic/inplace_pass.md)的input_inplace_pass配置





## 调用示例

```python
import torch
from torch._functorch.aot_autograd import aot_module_simplified
import torch_npu

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x = x + y
        return x

# 构建自定义的backend
def custom_backend(gm: torch.fx.GraphModule, example_inputs):
    compiler = torch.npu.npugraph_ex.compile_fx()
    return aot_module_simplified(gm, example_inputs, fw_compiler=compiler)
                        
x = torch.ones([2, 2], dtype=torch.int32).npu()
y = torch.ones([2, 2], dtype=torch.int32).npu()
model = torch.compile(Model().npu(), backend=custom_backend, fullgraph=True, dynamic=False)
ret = model(x, y)
```

