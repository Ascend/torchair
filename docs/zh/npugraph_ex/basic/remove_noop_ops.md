# 冗余算子消除功能

## 功能简介

aclgraph图模式场景下，npugraph\_ex集成了冗余算子消除优化功能，能够自动识别并消除计算图中不影响程序逻辑或数据计算的冗余操作。这种优化可以有效减少不必要的计算开销，提高模型执行效率。当与其它图优化策略结合使用时，可通过优化对比来选择最佳方案。

典型冗余操作示例如下：

- 无实际意义的张量视图操作（如b=tensor\_a\[:\]）
- 参数无效的特殊算子（如重复次数为1的repeat操作）

本功能**依赖PyTorch 2.2.0或更高版本**，不同版本支持的优化场景可能存在差异。以PyTorch 2.5.1为例，支持优化的算子包括但不限于下表，算子的介绍请参见PyTorch源码。

|算子名|冗余操作场景示例|
|--|--|
|aten.slice|对整个张量进行完整切片操作，如tensor_a[:]。|当算子的输入/输出Shape不一致，或优化后在输入/输出间引入了新的别名关系时，不会进行冗余消除操作。|
|aten.slice_scatter|对整个张量进行完整切片操作，如tensor_a.slice_scatter(tensor_b)。|
|aten.repeat|张量在所有待重复维度上重复的次数为1，如tensor_a.repeat(1)。|
|aten.constant_pad_nd|张量在所有待扩充维度上扩充的数量为0，如torch.nn.functional.pad(tensor_a, pad=[0, 0, 0, 0], value=3.5)。|
|torch.ops.prims.convert_element_type|张量数据类型转换时，前后一致。|
|torch.ops.prims.device_put|张量数据类型转换时，前后一致。|
|aten.ceil、aten.floor、aten.round、aten.trunc|张量数据类型为整型。|
|aten.pow|张量指数运算时幂为1。|
|aten.cat|张量拼接时，参与拼接的张量只有自身，如torch.cat([tensor_a])。|
|aten.view.default、aten.view.dtype|-|
|aten.copy、aten.alias、aten.clone|-|

非冗余操作场景下，当算子的输入/输出Shape不一致，或优化后在输入/输出间引入了新的别名关系时，不会进行冗余消除操作。

## 使用约束

本功能支持的产品型号参见[使用说明](../../overview.md#使用说明)。

## 使用方法

该功能通过npugraph\_ex的options配置，示例如下，仅供参考不支持直接拷贝运行，参数说明参见下表。

```python
import torch
import torch_npu

torch.compile(model, backend="npugraph_ex", options={"remove_noop_ops": True}, dynamic=False, fullgraph=True)
```

**表 1**  参数说明

|**参数名**|**参数说明**|
|--|--|
|remove_noop_ops|是否对冗余Kernel进行优化处理。True（默认值）：对冗余Kernel进行优化处理。False：不对冗余Kernel进行优化处理。|

## 使用说明

以“对整个张量进行完整切片操作”为例，当不对冗余Kernel进行优化时，计算图如下：

```txt
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %slice_1 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg0_1, 0, 0, 9223372036854775807), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_1, %arg1_1), kwargs = {})
    return (add,)
```

在本功能设置成功后，参考[图编译Debug信息保存功能](../dfx/debug_save.md)，在Debug信息的npugraph\_ex目录中的debug.log文件中可以看到优化后的计算图，如下：

```txt
after fx graph optimization, graph is graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
    return (add_3,)
```

可见冗余aten.slice操作被消除。
