# use\_internal\_format\_weight

## 功能说明

将模型中的权重weight转成TorchAir定义的内部私有格式。

## 函数原型

```python
use_internal_format_weight(model: torch.nn.Module) -> None
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| model | 输入 | 用户自定义的模型，继承原生的torch.nn.Module类。 | 是 |

## 返回值说明

无

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

model = Model()
torchair.use_internal_format_weight(model)
```

