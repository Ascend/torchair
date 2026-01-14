# Cast

## 功能说明

算子converter中的构图元素，表示一个Cast节点，即图中Tensor的类型转换方法。

## 函数原型

```python
Cast(x: Tensor, *, dst_type: int, dependencies=[], node_name=None) -> Tensor
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| x | 输入 | 待转换的Tensor。 | 是 |
| dst_type | 输入 | Tensor转换后的数据类型，类型取值参见[DataType类](DataType类.md)。 | 是 |
| dependencies | 输入 | 用于指定节点的控制边，标识该节点在图中的拓扑依赖关系，传入列表中为Tensor类型。 | 否 |
| node_name | 输入 | 节点名，默认为None，例如'const_1'，同一张图中节点名不允许重复。 | 否 |
| * | 输入 | 预留参数项，用于后续功能扩展。 | 否 |

## 返回值说明

正常情况下，返回新类型的Tensor，否则失败报错。

## 约束说明

本接口仅适用于max-autotune模式。

## 调用示例

```python
import torch_npu, torchair
from torchair.ge import DataType
Cast(1., dst_type=DataType.DT_INT64)
```
