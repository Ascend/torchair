# compile\_fx

## 功能说明

返回编译后的可执行FX图对象，可直接调用执行。

## 函数原型

```python
compile_fx(gm, example_inputs=None, options=None)
```

## 参数说明

|**参数**|**输入/输出**|**说明**|
|--|--|--|
|gm|输入|表示AOT（Ahead-of-Time）编译后的GraphModule类对象，gm.graph为其FX图。|
|example_inputs|输入|模型的示例输入。|
|options|输入|模型编译的功能配置项。|

## 返回值说明

返回编译后的可执行FX图对象。

## 约束说明

使用compile_fx接口构建自定义后端时，不支持以下基于npugraph_ex后端实现的功能：

- [多流表达功能](../../advanced/multi_stream.md)
- [AI-Core和Vector-Core限核功能](../../advanced/limit_cores.md)
- [FX图优化Pass配置功能](../../basic/inplace_pass.md)的input_inplace_pass配置需要配合`aot_module_simplified(keep_inference_input_mutations=True)`使用


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

# 构建自定义的compiler
def custom_compiler(gm: torch.fx.GraphModule, example_inputs):
    test_options = {
        "clone_input": False
    }
    compiled_graph = torch.npu.npugraph_ex.compile_fx(gm, example_inputs, test_options)
    return compiled_graph

# 构建自定义的backend
def custom_backend(gm: torch.fx.GraphModule, example_inputs):
    return aot_module_simplified(gm, example_inputs, fw_compiler=custom_compiler)
                        
x = torch.ones([2, 2], dtype=torch.int32).npu()
y = torch.ones([2, 2], dtype=torch.int32).npu()
model = torch.compile(Model().npu(), backend=custom_backend, fullgraph=True, dynamic=False)
ret = model(x, y)
```
