# 静态Kernel编译配置

## 功能简介

>**须知：** reduce-overhead模式下的功能为试验特性，后续版本可能存在变更，暂不支持应用于商用产品中。

对于纯静态shape网络或者shape变化较少的动态shape网络，如需提升网络执行性能，可通过算子预先静态编译达到目的，该方式简称为静态Kernel编译。它是指在模型编译时指定shape大小，运行时不需要指定shape大小，减少运行时开销，具体优势如下：

-   编译时已知所有Tensor的大小，存储空间利用率高。
-   编译时可以针对实际的shape大小做针对性优化。
-   AI处理器擅长并行指令运行，不擅长逻辑计算，如果有太多的Scalar操作可能会打断并行指令的运行，从而导致性能下降。静态编译可以在编译时完成标量的计算，一定程度上可以提升性能。
-   编译工具在编译时知道确切的操作数据大小，不会额外插入同步，不会导致并行执行多个指令变成串行执行，一定程度上可以提升性能。

开启静态Kernel编译后，系统根据输入的算子信息统计文件，得到确定的shape信息，针对每一个shape都编译出一个算子二进制。

## 使用约束

-   本功能支持如下产品：
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
    
-   本功能仅支持reduce-overhead模式。

-   需要注意的是，当Ascend PyTorch Profiler中**“experimental\_config”参数**开启算子信息统计功能（即record\_op\_args=True），且“**schedule**”**参数**设置的预先跳过的step轮数为0时（即skip\_first=0），不支持同时使用本功能。

    > **说明：** 
    >**Ascend PyTorch Profiler**是CANN针对PyTorch框架开发的性能分析工具，通过在PyTorch脚本中添加**Ascend PyTorch Profiler接口**（推荐torch\_npu.profiler.profile接口）采集指定指标和性能数据，详细的使用方法请参考《CANN 性能调优工具用户指南》中的“Ascend PyTorch Profiler”章节。

-   多卡场景使用限制：必须配置环境变量**[LOCAL\_WORLD\_SIZE](https://docs.pytorch.org/docs/stable/elastic/run.html#definitions)**（指单个物理节点（一台机器）上启动的并行进程数，常见于PyTorch多节点场景），其每个节点的LOCAL\_WORLD\_SIZE必须一致。

    > **说明：** 
    >由于多卡场景在安装静态kernel run包时，会更新算子库公共文件，若没有环境变量LOCAL\_WORLD\_SIZE的协同，可能会读取到过程态的内容，从而导致未定义行为（undefined behavior）。因此，在多卡场景使用该功能时，必须配置LOCAL\_WORLD\_SIZE。

## 使用方法

1.  （可选）配置环境变量。仅多卡场景需要配置。

    ```bash
    export LOCAL_WORLD_SIZE=${local_world_size}
    ```

    $\{local\_world\_size\}表示单机上实际运行的进程数，取值是自然数。

2.  通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config配置本功能，示例如下，仅供参考不支持直接拷贝运行，参数介绍参见下表。

    ```python
    import torch_npu, torchair
    config = torchair.CompilerConfig()
    # 配置图执行模式
    config.mode = "reduce-overhead"
    # 开启静态Kernel编译
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "/path/test"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    opt_model = torch.compile(model, backend=npu_backend)
    ```

    **表 1**  参数说明

    
    | 参数名 | 说明 |
| --- | --- |
| _aclnn_static_shape_kernel | 布尔类型，是否开启静态Kernel编译 。<br>- False（默认值）：默认关闭。<br>- True：开启静态Kernel编译 。 |
| _aclnn_static_shape_kernel_build_dir | 字符串类型，配置编译产物路径，默认为当前执行脚本的同级目录。<br>说明： 请确保该参数指定的路径确实存在，并且运行用户具有读、写操作权限。 |

## 产物说明

假设产物路径为“/path/test”，目录结构如下，其中$\{timestamp\}为时间戳、$\{pid\}表示运行的进程号。

```
aclnn_static_shape_kernel_outputs               // 固定的产物文件名
|—— ${timestamp_1}_${pid_1}_outputs 
|    |—— ${pid_1}                                // 模型中目标算子信息文件夹
|        |—— MatMulV2_float_ND_1_2048_0.json     // js文件表示网络中的算子统计信息，包括shape和format等
|        |—— ......json
|    |—— ${pid_1}_debug                          // 模型中全量算子信息文件夹
|        |—— MatMulV2_float_ND_1_2048_0.json
|        |—— ......json
|    |—— ${pid_1}_opcompile                      // 模型中支持静态编译的算子信息导出目录
|        |—— MatMulV2_float_ND_1_2048_0.json     
|        |—— ......json
|    |—— static_kernel_${datetime}.run         // 编译好的静态Kernel文件
|—— ${timestamp_2}_${pid_2}_outputs 
|—— ......
```

