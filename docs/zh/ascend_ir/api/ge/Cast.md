# Cast

## 功能说明

算子Converter中的构图元素，表示一个Cast节点，即图中Tensor的类型转换方法。

## 函数原型

```python
Cast(x: Tensor, *, dst_type: int, dependencies=[], node_name=None) -> Tensor
```

## 参数说明

|参数|输入/输出|说明|
|--|--|--|
|x|输入|待转换的Tensor。|
|dst_type|输入|Tensor转换后的数据类型，类型取值参见DataType类。|
|dependencies|输入|用于指定节点的控制边，标识该节点在图中的拓扑依赖关系，传入列表中为Tensor类型。|
|node_name|输入|节点名，默认为None，例如'const_1'，同一张图中节点名不允许重复。|
|*|输入|预留参数项，用于后续功能扩展。|

## 返回值说明

正常情况下，返回新类型的Tensor，否则失败报错。

## 约束说明

无

## 调用示例

```python
import torch_npu, torchair
from torchair.ge import DataType
Cast(1., dst_type=DataType.DT_INT64)
```
