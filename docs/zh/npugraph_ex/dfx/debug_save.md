# 图编译Debug信息保存功能

## 功能简介

为方便问题定位过程中的信息收集，npugraph\_ex通过复用PyTorch**原生DEBUG环境变量TORCH\_COMPILE\_DEBUG**，当其设置为1时，将自动开启所有必要的日志打印与文件Dump。

**图 1**  图编译示意图  
![](../../figures/graph_compile_1.png "图编译示意图")

开启本功能后，图编译过程中能自动收集的关键调试信息如上图所示 ，详细说明参见下表。

**表 1**  信息收集表

|信息类型|说明|
|--|--|
|日志信息|PyTorch原生Dynamo日志<br>npugraph\_ex 日志|
|Debug信息|AOT前的GraphModule<br>AOT后的GraphModule<br>每个公共Pass处理后的FX图（txt文件）<br>aclgraph优化中不同Pass处理后的FX图（txt文件）<br>aclgraph编译后的FX图结构信息（output_code.py文件）<br>aclgraph在Capture阶段捕获的算子执行图信息（*.json文件）<br>注意：仅当配套的CANN版本是8.5.0及之后的版本，才会有该文件生成，否则不会生成。|

## 使用约束

- 使用[compile_fx](../api/npugraph_ex//compile_fx.md)开启该功能时，仅收集编译流程中的部分调试产物。
- 本功能支持的产品型号参见[使用说明](../../overview.md#使用说明)。

## 使用方法

在执行脚本前，先在终端设置环境变量：

```bash
export TORCH_COMPILE_DEBUG=1
python main.py
```

假设待执行的脚本main.py代码如下，该脚本的目标是实现简单的乘法。

1. 开启Dynamo日志。
2. 设置图编译模式。
3. 构造一个随机输入，设置dynamic=False及requires\_grad=False，执行了一次推理。

    ```python
    import logging
    import torch
    import torch_npu
    
    # 开启Dynamo日志
    torch._logging.set_logs(dynamo=logging.DEBUG, aot=logging.DEBUG, output_code=True, graph_code=True) 
    class Model(torch.nn.Module):
        def forward(self, x):
            return 2 * x
    
    model = Model().npu()
    model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    x = torch.randn(20, 20, requires_grad=False).npu()  
    out = model(x)
    ```

    运行示例脚本，编译过程中必要的Debug信息产物目录结构如下，仅供参考，具体取决于实际开启的Pass数量。“torch\_compile\_debug”为PyTorch原生开启环境变量时创建的目录，默认在当前脚本路径下。

    ```txt
    torch_compile_debug/run_<时间>-pid_<进程号>
    ├── npugraph\_ex
    │   ├── debug.log                                                 # npugraph\_ex日志
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
