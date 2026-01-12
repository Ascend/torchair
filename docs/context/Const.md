# Const

## 功能说明

算子converter中的构图元素，表示一个Const节点，即图中的常量值。

该常量值在构建计算图时定义，且在整个图的执行过程中不会改变。

## 函数原型

```python
Const(v: Any, dtype: int = None, node_name=None, readable=True) -> Tensor
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| v | 输入 | 设定的常量值，支持任意数据类型，例如1.0。 | 是 |
| dtype | 输入 | 常量值的数据类型，默认为整型，类型取值参见[DataType类](DataType类.md)。 | 否 |
| node_name | 输入 | 常量节点名，默认为None，例如'const_1'，同一张图中节点名不允许重复。 | 否 |
| readable | 输入 | 是否在图上增加属性以可读的方式记录const值，默认为True。<br>- True：开启可读方式记录常量值。<br>- False：不开启可读方式记录常量值。 | 否 |

## 返回值说明

正常情况下，返回常量Tensor，否则失败报错。

## 约束说明

本接口仅适用于max-autotune模式。

## 调用示例

```python
import torch_npu, torchair
from torchair.ge import DataType
Const(0, dtype=DataType.DT_INT64, node_name='Const0')
```
