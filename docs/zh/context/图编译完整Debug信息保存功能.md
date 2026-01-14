# 图编译完整Debug信息保存功能

## 功能介绍

TorchAir目前提供了多种定位方法，包括日志和文件dump等，这些方法通常通过单独的开关或配置项来开启。在问题定位过程中，用户需要参考不同文档，分别进行配置和手动收集信息。尽管该方式具有一定灵活性，但也增加了操作的复杂性。

本章功能旨在简化模型问题定位过程中的信息收集，通过复用PyTorch**原生DEBUG环境变量TORCH\_COMPILE\_DEBUG**，当其设置为1时，将自动开启所有必要的日志打印与文件dump，实现快速收集问题信息的目的。

**图 1**  图编译过程示意图 

![](figures/图编译过程示意图.png "图编译过程示意图")

开启本功能后，图编译过程中能自动收集的关键调试信息如上图所示 ，详细说明参见下表。

**表 1**  Debug信息项


| 场景分类 | Debug信息 |
| --- | --- |
| 公共Debug信息 |  - 日志信息收集：PyTorch原生Dynamo日志、[TorchAir Python层日志](TorchAir-Python层日志.md)、[TorchAir C++层日志](TorchAir-C++层日志.md)<br>- ①AOT前的GraphModule<br>- ②AOT后的GraphModule<br>- ③公共Pass图优化过程中每个Pass的输出FX图（txt文件） |
| max-autotune模式Debug信息 | - **公共Debug信息**<br>- ④GE图优化前后及不同Pass处理后的图结构信息：该图结构信息可通过config.debug.graph_dump.type设置txt、pbtxt、py文件类型，可参考[图结构dump功能](图结构dump功能.md)。 |
| reduce-overhead模式Debug信息 | - **公共Debug信息**<br>- ⑤aclgraph优化中不同Pass处理后的FX图（txt文件）<br>- ⑥aclgraph在Capture阶段捕获的算子执行图信息（*.json文件）：注意仅当配套的CANN商发版本是8.5及之后的版本，才会有该文件生成，否则不会生成。<br>- ⑦aclgraph编译后的FX图结构信息（output_code.py文件） |

## 使用方法

-   方法一：在终端设置环境变量

    ```bash
    export TORCH_COMPILE_DEBUG=1
    python main.py
    ```

-   方法二：在Python脚本开头设置环境变量

    > **说明：** 
    >若在import torchair及from torchair import logger之后设置环境变量，会因环境变量未在日志模块导入前生效，导致目录下缺失torchair/debug.log文件。

    ```python
    import os
    # 配置环境变量
    os.environ["TORCH_COMPILE_DEBUG"] = "1" 
    import torch
    import torch.nn as nn
    import torchair
    from torchair import logger
    ```

## 使用示例（max-autotune模式）

以如下模型脚本为例，实现了简单的乘法功能，代码流程如下：

1.  设置环境变量TORCH\_COMPILE\_DEBUG为1。
2.  开启Dynamo日志。
3.  设置图编译模式，设置device="npu:0"。
4.  构造一个随机输入，设置dynamic=False及requires\_grad=True，执行了一次前向和反向。
5.  构造一个不同shape的随机输入，设置requires\_grad=False，执行了一次前向推理。

```python
import os
# 配置环境变量
os.environ["TORCH_COMPILE_DEBUG"] = "1"  
import torch
import torch_npu
import torchair   
import logging
# 开启Dynamo日志
torch._logging.set_logs(dynamo=logging.DEBUG,aot=logging.DEBUG,output_code=True,graph_code=True) 

config = torchair.CompilerConfig()
# 仅适用max-autotune模式
config.debug.graph_dump.type = "pbtxt"   
config.mode.value = "max-autotune"
npu_backend = torchair.get_npu_backend(compiler_config=config)
device = "npu:0"

class Model(torch.nn.Module):
    def forward(self, x):
        return 2 * x

model = Model().to(device)
model = torch.compile(model, backend=npu_backend, dynamic=False)

x = torch.randn(10, 10, requires_grad=True, device=device)
out = model(x)                
loss_fn = torch.nn.MSELoss()
target = torch.randn(10, 10, device=device)
loss = loss_fn(out, target)
loss.backward()

x = torch.randn(20, 20, requires_grad=False, device=device)  
out = model(x)
```

运行示例脚本，编译过程中必要的Debug信息产物目录结构如下，仅供参考，具体取决于实际开启的Pass数量。“torch\_compile\_debug”为PyTorch原生开启环境变量时创建的目录，默认在当前脚本路径下。

```bash
torch_compile_debug/run_<时间>-pid_<进程号>
├── torchair
│   ├── debug.log                                                    # Python和C++层日志
│   ├── model__0                                                     # model__0为模型ID
│   │   ├── backward                                                 # 反向编译
│   │   │   ├── 000_aot_backward_graph.txt                           # AOT后的GraphModule
│   │   │   ├── 001_aot_backward_graph_after_${pass1_name}.txt       # 公共图优化过程中每个Pass的输出FX图
│   │   │   ├── 002_aot_backward_graph_after_${pass2_name}.txt
│   │   │   ├── 003_aot_backward_original_ge_graph.pbtxt             # 所有GE图优化处理前的GE图
│   │   │   ├── 004_aot_backward_graph_after_${pass3_name}.pbtxt     # GE图优化中不同Pass处理后的GE图
│   │   │   ├── 005_aot_backward_graph_after_${pass4_name}.pbtxt  
│   │   │   ├── 006_aot_backward_optimized_ge_graph.pbtxt            # 所有GE图优化处理后的GE图
│   │   │   ├── ......                                               # 其他Pass优化
│   │   ├── dynamo_out_graph.txt                                     # AOT前的GraphModule
│   │   ├── forward                                                  # 前向编译
│   │   │   ├── 000_aot_forward_graph.txt   
│   │   │   ├── 001_aot_forward_graph_after_${pass1_name}.txt
│   │   │   ├── 002_aot_forward_graph_after_${pass2_name}.txt
│   │   │   ├── 003_aot_forward_original_ge_graph.pbtxt
│   │   │   ├── 004_aot_forward_graph_after_${pass3_name}.pbtxt 
│   │   │   ├── 005_aot_forward_graph_after_${pass4_name}.pbtxt
│   │   │   ├── 006_aot_backward_optimized_ge_graph.pbtxt
│   ├── model__1
│   │   ├── dynamo_out_graph.txt
│   │   ├── forward                                                # 前向推理
│   │   │   ├── 000_aot_forward_graph.txt   
│   │   │   ├── 001_aot_forward_graph_after_${pass1_name}.txt
│   │   │   ├── 002_aot_forward_graph_after_${pass2_name}.txt
│   │   │   ├── 003_aot_forward_original_ge_graph.pbtxt
│   │   │   ├── 004_aot_forward_graph_after_${pass3_name}.pbtxt 
│   │   │   ├── 005_aot_forward_graph_after_${pass4_name}.pbtxt
│   │   │   ├── 006_aot_forward_optimized_ge_graph.pbtxt
└── torchdynamo
    └── debug.log     # Torch原生dynamo日志
```

## 使用示例（reduce-overhead模式）

以如下模型脚本为例，实现了简单的乘法功能，代码流程如下：

1.  设置环境变量TORCH\_COMPILE\_DEBUG为1。
2.  开启Dynamo日志。
3.  设置图编译模式，设置device="npu:0"。
4.  构造一个随机输入，设置dynamic=False及requires\_grad=False，执行了一次推理。

```python
import os
# 配置环境变量
os.environ["TORCH_COMPILE_DEBUG"] = "1"  
import torch
import torch_npu
import torchair   
import logging
# 开启Dynamo日志
torch._logging.set_logs(dynamo=logging.DEBUG,aot=logging.DEBUG,output_code=True,graph_code=True) 

config = torchair.CompilerConfig()
config.mode.value = "reduce-overhead"
npu_backend = torchair.get_npu_backend(compiler_config=config)
device = "npu:0"

class Model(torch.nn.Module):
    def forward(self, x):
        return 2 * x

model = Model().to(device)
model = torch.compile(model, backend=npu_backend, dynamic=False)

x = torch.randn(20, 20, requires_grad=False, device=device)  
out = model(x)
```

运行示例脚本，编译过程中必要的Debug信息产物目录结构如下，仅供参考，具体取决于实际开启的Pass数量。“torch\_compile\_debug”为PyTorch原生开启环境变量时创建的目录，默认在当前脚本路径下。

```bash
torch_compile_debug/run_<时间>-pid_<进程号>
├── torchair
│   ├── debug.log                                                 # Python和C++层日志
│   ├── model__0                                                  # model__0为模型ID
│   │   ├── forward                                               # 前向推理
│   │   │   ├── output_code.py                                    # 编译后的图结构文件
│   │   │   ├── 000_aot_forward_graph.txt                         # AOT后的GraphModule
│   │   │   ├── 001_aot_forward_graph_after_${pass1_name}.txt     # 公共图优化过程中每个Pass的输出FX图
│   │   │   ├── 002_aot_forward_graph_after_${pass2_name}.txt
│   │   │   ├── 003_aot_forward_graph_after_${pass5_name}.txt     # aclgraph优化中不同pass处理后的FX图
│   │   │   ├── 004_aot_forward_graph_after_${pass6_name}.txt  
│   │   │   ├── ......                                            # 其他Pass优化
│   │   ├── dynamo_out_graph.txt                                  # AOT前的GraphModule
│   │   ├── graph_1_id_${aclgraph_id}_rank_${rank_id}_pid_${pid}_ts_${timestamp}.json      # 捕获的算子执行图信息
└── torchdynamo
    └── debug.log                                                 # Torch原生Dynamo日志
```

