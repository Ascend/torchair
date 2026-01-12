# register\_fx\_node\_ge\_converter

## 功能说明

将自定义算子注册到TorchAir框架中。

## 函数原型

```python
register_fx_node_ge_converter(aten_op)
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| aten_op | 输入 | 待注册的算子名，例如torch.ops.aten.xxx。 | 是 |

## 返回值说明

返回值用作装饰器，无法独立使用。

## 约束说明

本接口仅适用于max-autotune模式。

## 调用示例

> **说明：** 
>简单示例如下，如需深入了解请参考[自定义算子入图](自定义算子入图.md)不同样例中“实现Converter”章节。

```python
import torch
import torch_npu
from torchair import register_fx_node_ge_converter
from torchair.ge import Tensor

# 装饰器，torch.ops.aten.sin.default为自定义算子的Python函数签名
@register_fx_node_ge_converter(torch.ops.aten.sin.default)
def converter_aten_sin_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sin(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.sin.default ge_converter is not implemented!")
```

