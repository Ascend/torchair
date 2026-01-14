# TensorSpec类

提供了TensorSpec定义，TensorSpec表示算子在Meta推导过程中得到的性能，当前主要用于算子入图的converter函数入参的类型声明。TensorSpec类具体定义如下：

```python
class TensorSpec():
    @abstractmethod
    def __init__(self):
        ...
    @abstractmethod
    def dtype():
        ...
    @abstractmethod
    def rank():
        ...
    @abstractmethod
    def size():
        ...
```

关于TensorSpec类中成员方法的介绍请参见下表。

**表 1**  TensorSpec类成员说明


| 属性名 | 属性说明 |
| --- | --- |
| \_\_init\_\_ | 构造方法。 |
| size | 描述Tensor的维度长度。 |
| dtype | Tensor的数据类型。 |
| rank | Tensor的dimension数。 |

