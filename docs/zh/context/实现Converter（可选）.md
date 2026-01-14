# 实现Converter（可选）

如果您希望使用[max-autotune模式功能](max-autotune模式功能.md)，例如SuperKernel等高阶能力，需要额外实现Ascend Converter（使用Ascend IR表达算子的计算逻辑）。

在Eager模式下，my\_inplace会调用Ascend C算子MyInplace；而对应到Converter实现，调用Ascend IR MyInplace。

> **说明：** 
>-   在Ascend C算子工程编译时，除了生成aclnnXxx接口外，还会同步生成同名Ascend IR的注册代码。
>-   接口介绍参见[register\_fx\_node\_ge\_converter](register_fx_node_ge_converter.md)和[custom\_op](custom_op.md)。

通常不需要手动实现Converter，TorchAir会自动完成PyTorch算子到同名（大驼峰）Ascend IR的转换。例如本样例中的my\_inplace算子，会自动转换为Ascend IR MyInplace。

如果自动转换无法完成，TorchAir的编译报错信息会给出原因，原因一般如下：

-   PyTorch算子的名字无法与Ascend IR名字通过大驼峰格式对应，例如my\_inplace实际对应的Ascend IR名字为MyInplace等。
-   PyTorch算子与Ascend IR的输入输出顺序或数量不一致。
-   PyTorch算子原型定义中存在Scalar类型入参。

您可以修改PyTorch算子原型使其满足条件，让TorchAir自动完成转换，或者手动实现Converter：

在third\_party/torchair/torchair/python/torchair/\_ge\_concrete\_graph/ge\_converter/custom目录下，新建MyInplace算子对应的my\_inplace.py文件，添加如下代码实现Converter：

```python
import torch
import torchair

# 实现原地算子的converter
@torchair.register_fx_node_ge_converter(torch.ops.npu.my_inplace.default)
def converter_my_inplace(x, y):   # 函数入参与Torch算子保持一致
    out = torchair.ge.custom_op("MyInplace", x, y)
    return out[1] # my_inplace算子只有一个输出z，所以这里是返回Ascend IR的第二个输出z
```

