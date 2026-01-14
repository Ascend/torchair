# Tensor类

提供了Tensor定义，用于算子入图的converter函数入参的类型声明。Tensor类具体定义如下：

```python
class Tensor():
    @abstractmethod
    def __init__(self):
        ...
    @abstractmethod
    def index(self):
        ...
    @abstractmethod
    def dtype():
        ...
    @abstractmethod
    def rank():
        ...
```

关于Tensor类中成员方法的介绍请参见下表。

**表 1**  Tensor类成员说明


| 属性名 | 属性说明 |
| --- | --- |
| \_\_init\_\_ | 构造方法。 |
| index | 描述Tensor属于算子的第几个输入或者输出。 |
| dtype | Tensor的数据类型。 |
| rank | Tensor的dimension数。 |

