# 算子插件化适配TorchAir图模式

## 非In-place算子插件化入图样例

假设您已经实现了[非In-place算子开发和入图样例](non_in_place_op_cases.md)中的PyTorch算子torch.ops.npu.my\_op，并能在Eager模式下正常调用。

```python
- func: my_op(Tensor x, Tensor? y, Tensor[] z, float attr1, int attr2) -> Tensor
```

插件化适配的Meta推导函数、Converter实现与上述示例完全一致，唯一的区别是可以将实现写在单独的py文件中，并在模型执行前加载。

完整的代码实现如下所示：

```python
import torch
import torch_npu
import torchair
from torch.library import Library
m = Library("npu", "IMPL", "Meta")  # 获取my_op算子所在的算子库

# 实现PyTorch算子的Meta推导
@torch.library.impl(m, "my_op")
def my_op_meta(x, y, z, attr1, attr2):
    return torch.empty_like(x)
# 仅当需要使用max-autotune模式，且对应Ascend C算子以算子工程开发时需要
@torchair.register_fx_node_ge_converter(torch.ops.npu.my_op.default)
def convert_npu_my_op(x, y, z, attr1, attr2):
    return torchair.ge.custom_op("MyOp", x, y, z, attr1, attr2)    
```

## In-place算子插件化入图样例

假设您已经实现了[In-place算子开发和入图样例](in_place_op_cases.md)中的PyTorch算子torch.ops.npu.my\_inplace，并能在Eager模式下正常调用。

```python
- func: my_inplace(Tensor(a!) x, Tensor y) -> Tensor
```

插件化适配时的Meta推导函数、函数化转换及Converter与示例实现完全一致，唯一的区别是可以将实现写在单独的py文件中，并在模型执行前加载。

完整的代码实现如下所示：

```python
import torch
import torch_npu
import torchair

m = torch.library.Library("npu", "FRAGMENT")

# 实现Inplace算子的Meta推导
@torch.library.impl(m, "my_inplace", "Meta")
def my_inplace_meta(x, y):
     return torch.empty_like(y)

# 仅当需要使用max-autotune模式，且对应Ascend C算子基于算子工程开发时需要
@torchair.register_fx_node_ge_converter(torch.ops.npu.my_inplace.default)
def converter_my_inplace(x, y):
    out = torchair.ge.custom_op(  # 根据算子定义设置变量名，顺序保持一致
        "MyInplace", # 使用原地算子的Ascend IR
        inputs={
            "x": x,
            "y": y,
        },
        outputs=['x', 'z'] # 原地算子的Ascend IR有两个输出x和z
    )
    return out[1] # my_inplace算子只有一个输出z，所以这里是返回Ascend IR的第二个输出z
```
