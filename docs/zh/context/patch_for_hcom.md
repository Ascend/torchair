# patch\_for\_hcom

## 功能说明

针对PyTorch 2.1版本，补齐PyTorch原生部分集合通信算子无法入图功能。

## 函数原型

```python
patch_for_hcom()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

无

## 调用示例

```python
import torch
import torch_npu
import torchair

# 在图执行之前调用patch方法
torchair.patch_for_hcom()

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return torch.add(x, y)
model = Model().npu()

config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)

# 执行编译后的Model
x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
opt_model(x, y)
```

