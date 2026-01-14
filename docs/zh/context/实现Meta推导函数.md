# 实现Meta推导函数

PyTorch原生要求所有能与torch.compile配合工作的算子需要实现Meta推导函数，又称为“符号化推导”。Meta函数表示了PyTorch算子输出与输入shape、dtype以及内存的关系，它是PyTorch入图的前提条件，借助符号化和符号guard可静态化控制流和形状信息，从而确定图结构。关于Meta函数的详细介绍请参考PyTorch官网[符号化手册](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.fh8zzonyw8ng)。

> **说明：** 
>
> -   Meta推导函数**必须在torch.compile执行前**完成注册。
> -   torch.library.Library接口介绍请参考[PyTorch官网](https://docs.pytorch.org/docs/stable/library.html#torch.library.Library)。

进入third\_party/op-plugin/op\_plugin/python/meta/\_meta\_registrations.py实现Meta推导函数：

```python
import torch
from torch.library import Library, impl

# meta register implementation
m = Library("npu", "IMPL", "Meta")
@impl(m, "my_inplace")
def my_inplace_meta(x, y):
    return torch.empty_like(y)            # 输出的shape、dtype与输入y相同
```

-   my\_inplace\_meta：Meta函数名，通常以PyTorch算子名+"\_meta"后缀命名。
-   m：表示NPU算子的Meta实现库，通常定义在文件开头“m=Library\("npu", "IMPL", "Meta"\)”。

