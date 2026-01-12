# dynamo\_export

## 功能说明

用于导出TorchAir生成的离线图（air格式），导出的图不再依赖PyTorch框架，可直接由CANN软件栈加载执行，减少框架调度带来的性能损耗，方便在不同的环境上部署移植，功能详情参见[Dynamo导图功能](Dynamo导图功能.md)。

## 函数原型

```python
dynamo_export(*args, model: torch.nn.Module, export_path: str = "export_file", export_name: str = "export", dynamic: bool = False, config=CompilerConfig(), **kwargs)
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| model | 输入 | 需要导出的模型，继承原生的torch.nn.Module类。 | 是 |
| export_path | 输入 | 离线图导出的文件存放路径，字符串类型，默认值为当前执行路径下的"export_file"目录。<br>**说明**：请确保该参数指定的路径确实存在，并且运行用户具有读、写操作权限。 | 否 |
| export_name | 输入 | 离线图导出的图名称，字符串类型，默认值为"export"。 | 否 |
| dynamic | 输入 | 设置导出静态模型还是动态模型，布尔类型。<br>- False（默认值）：导出静态模型。<br>- True：导出动态模型。 | 否 |
| config | 输入 | 图编译配置，[CompilerConfig类](CompilerConfig类.md)的实例化，默认情况下采用TorchAir自动生成的配置。<br>导图时支持配置auto_atc_config_generated和enable_record_nn_module_stack功能，配置样例参见[使用方法](Dynamo导图功能.md#使用方法)。 | 否 |
| *args、**kwargs | 输入 | 导出模型时的样例输入，不同的输入可能导致模型执行不同的分支，进而导致trace的图不同。应当选取执行推理时的典型值。 | 否 |

## 返回值说明

无

## 约束说明

-   本功能仅支持max-autotune模式，暂不支持同时配置[固定权重类输入地址功能（Ascend IR）](固定权重类输入地址功能（Ascend-IR）.md)。
-   导出时需要保证被导出部分能构成一张图。
-   支持单卡和多卡场景下导出图，且支持导出后带AllReduce等通信类算子。
-   导出的air文件大小不允许超过2G（依赖的第三方库Protobuf存在限制导致）。
-   受Dynamo功能约束，不支持动态控制流if/else。

## 调用示例

以单卡场景下的导图过程为例，代码如下，导图结果和多卡场景下的导图示例请参见[使用示例](Dynamo导图功能.md#使用示例)。

```python
import torch
import torch_npu
import torchair

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.p1 = torch.nn.Parameter(torch.randn(2, 4))
        self.p2 = torch.nn.Parameter(torch.randn(2, 4))

    def forward(self, x, y):
        x = x + y + self.p1 + self.p2
        return x

model = Model()
x = torch.randn(2, 4)
y = torch.randn(2, 4)
torchair.dynamo_export(x, y, model=model, export_path="./test_export_file_False", dynamic=False)
```

