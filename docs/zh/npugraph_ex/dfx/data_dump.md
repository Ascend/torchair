# 算子Data Dump功能

## 功能简介

本功能可以Dump aclgraph模式下整图执行时的图输入、每个算子的输出数据，用于后续问题定位和分析，如算子运行性能或精度问题。

## 使用约束

本功能支持的产品型号参见[使用说明](../../overview.md#使用说明)。

## 使用方法

场景1：通过npugraph\_ex的options配置，Dump整图算子。示例如下，参数说明参见下表。

```python
import torch
model = torch.compile(model, backend="npugraph_ex", options={
    "dump_tensor_data": True, 
    "data_dump_dir": "/home/test/dump/",
    "data_dump_stage": "optimized"
    }, dynamic=False)
```

**表 1**  参数说明

|**参数名**|**参数说明**|
|--|--|
|dump_tensor_data|是否开启数据dump功能，bool类型。False（默认值）：不开启数据dump。True：开启数据dump。|
|data_dump_stage|dump数据阶段，字符串类型。<br>original：dump最原始的、未经过TorchAir优化的fx图。注意，开启该模式后，会通过eager mode只执行最原始的fx图，不会进入aclgraph模式。<br>optimized（默认值）：dump经过TorchAir优化后的fx图。|
|data_dump_dir|dump数据的存放路径，字符串类型，默认值为当前执行路径。支持配置绝对路径或相对路径（相对执行命令行时的当前路径）。该路径需要是确实存在，并且运行用户具有读、写操作权限绝对路径配置以“/”开头，例如：/home/HwHiAiUser/output。相对路径配置直接以目录名开始，例如：output。|

执行成功后，在当前执行路径或data\_dump\_dir指定的目录下生成文件夹，如：worldsize1\_global\_rank0，多卡场景会生成多个文件夹。

```txt
|—— add_device_0_0.pt         // 算子的输出
|—— arg0_1_device_0_0.pt        // 图的输入
|—— getitem_device_0_0.pt       // 以getitem开头的文件表示多输出算子的输出
```

上述结果文件中，device后的第一个0表示设备序号，第二个0表示计数标识。可以通过torch.load查看或者使用Netron软件查看，产物命名与fx图上的节点名称一一对应。

场景2：通过**torch\_npu.save\_npugraph\_tensor**接口Dump单个算子输入、输出信息。

该接口提供了类似原生print特性且不影响aclgraph replay的tensor dump能力，允许将图内计算节点的tensor数据、数据类型、shape信息保存到指定的pt或bin文件中以便观察aclgraph的执行过程。可以使用torch.load接口读取保存的tensor。

该接口详细介绍参考《Ascend Extension for PyTorch 自定义 API参考》中的“torch\_npu.save\_npugraph\_tensor”章节。

```python
import torch
import torch_npu

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x0, x1):
        torch_npu.save_npugraph_tensor(x1, save_path="./test1.pt")
        sq1 = torch.square(x0)
        torch_npu.save_npugraph_tensor(sq1, save_path="./test2.pt")
        add1 = torch.add(x1, sq1)
        mm1 = torch.mm(x0, add1)
        return mm1

x0 = torch.randn([10, 10]).npu()
x1 = torch.randn([10, 10]).npu()
model = Model()
model = torch.compile(model, backend="npugraph_ex"
)
output = model(x0, x1)

```

输出结果说明：

```txt
test1_device_0_0.pt   // 输入x1的产物
test2_device_0_0.pt   // 算子pow的输出产物
```

上述结果文件中，device后的第一个0表示设备序号，第二个0表示计数标识。可以通过torch.load查看或者使用Netron软件查看，产物命名与脚本中传入的save\_path参数对应。

