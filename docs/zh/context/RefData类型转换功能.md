# RefData类型转换功能

## 功能简介

在大模型推理场景下，如果存在Ref类算子（例如Assign、ScatterUpdate等算子，类似于PyTorch中的in-place类算子）改写输入内存的情况，可以在构图过程中将用户输入的Data类型转换为RefData类型，以减少重复数据拷贝，提高模型执行效率。

## 使用约束

本功能仅支持max-autotune模式。

## 使用方法

-   **对于离线推理场景**

    先使用[Dynamo导图功能](Dynamo导图功能.md)导出离线图再进行后续AI应用开发，该场景下默认已开启RefData数据类型转换功能。

-   **对于在线推理场景**

    先使用torch.compile进行图编译，再进行图执行。此场景下需通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数说明如下表。

    ```python
    import torch_npu
    import torchair
    config = torchair.CompilerConfig()
    # 使能RefData类型的开关
    config.experimental_config.enable_ref_data = True
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    opt_model = torch.compile(model, backend=npu_backend)
    ```

    **表 1**  参数说明
    
    | 参数名 | 说明 |
	| --- | --- |
	| enable_ref_data | 构图过程中是否将输入数据类型转换为RefData类型。<br>- False（默认值）：不转换为RefData类型。<br>- True：转换为RefData类型。 |

## 使用示例

以在线推理场景为样例，示例代码如下：

```python
import torch
import torch_npu
import torchair 
from torch import nn
from torchair.configs.compiler_config import CompilerConfig

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
    def forward(self, x):
        return x.add_(1)

device = torch.device("npu:0")
config = CompilerConfig()
config.experimental_config.enable_ref_data = True
input0 = torch.ones((3,3), dtype=torch.float32)
input0 = input0.to(device)
model = Network()
npu_backend = torchair.get_npu_backend(compiler_config=config)
model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=True)
```

使能RefData数据类型转换功能后，开启[TorchAir Python层日志](TorchAir-Python层日志.md)，有如下打印信息：

```
[DEBUG] TORCHAIR 20240607 02:06:15 Replace RefData_5_3_20_20_1200_400_20_1_0_140251860631280:RefData with arg0_1:Data in graph graph_1
```

