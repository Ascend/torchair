# 静态Kernel编译功能

## 功能简介

对于纯静态shape网络或者shape变化较少的动态shape网络，如需提升网络执行性能，可通过算子预先静态编译达到目的，该方式简称为静态Kernel编译。它是指在模型编译时指定shape大小，运行时不需要指定shape大小，减少运行时开销，具体优势如下：

-   编译时已知所有Tensor的大小，存储空间利用率高。
-   编译时可以针对实际的shape大小做针对性优化。
-   AI处理器擅长并行指令运行，不擅长逻辑计算，如果有太多的Scalar操作可能会打断并行指令的运行，从而导致性能下降。静态编译可以在编译时完成标量的计算，一定程度上可以提升性能。
-   算子编译工具（[op\_compiler](https://hiascend.com/document/redirect/CannCommunityopcompiler)）在编译时知道确切的操作数据大小，不会额外插入同步，不会导致并行执行多个指令变成串行执行，一定程度上可以提升性能。

开启静态Kernel编译后，系统根据输入的算子信息统计文件，得到确定的shape信息，针对每一个shape都编译出一个算子二进制。

## 使用约束

-   本功能支持的产品型号参见[使用说明](../../overview.md#使用说明)。

-   需注意的是，当Ascend PyTorch Profiler中**“experimental\_config”参数**开启算子信息统计功能（即record\_op\_args=True），且“**schedule**”**参数**设置的预先跳过的step轮数为0时（即skip\_first=0），不支持同时使用本功能。

    > [!NOTE]说明
    >**Ascend PyTorch Profiler**是CANN针对PyTorch框架开发的性能分析工具，通过在PyTorch脚本中添加**Ascend PyTorch Profiler接口**（推荐torch\_npu.profiler.profile接口）采集指定指标和性能数据，详细的使用方法请参考《CANN 性能调优工具用户指南》中的“Ascend PyTorch调优工具”章节。

-   多卡场景使用限制：必须配置环境变量[**LOCAL\_WORLD\_SIZE**](https://docs.pytorch.org/docs/stable/elastic/run.html#definitions)（指单个物理节点（一台机器）上启动的并行进程数，常见于PyTorch多节点场景），其每个节点的LOCAL\_WORLD\_SIZE必须一致。

    > [!NOTE]说明
    >-   由于多卡场景在安装静态kernel run包时，会更新算子库公共文件，若没有环境变量LOCAL\_WORLD\_SIZE的协同，可能会读取到过程态的内容，从而导致未定义行为（undefined behavior）。因此，在多卡场景使用该功能时，必须配置LOCAL\_WORLD\_SIZE。

## 使用方法

1.  （可选）配置环境变量。仅多卡场景需要配置。

    ```bash
    export LOCAL_WORLD_SIZE=${local_world_size}
    ```

    $\{local\_world\_size\}表示单机上实际运行的进程数，取值是自然数。

2.  通过npugraph\_ex的options配置，示例如下，仅供参考不支持直接拷贝运行，参数说明参见下表。

    ```python
    import torch
    import torch_npu
    
    opt_model = torch.compile(model, backend="npugraph_ex", options={"static_kernel_compile": True}, dynamic=False, fullgraph=True)
    ```

**表 1**  参数说明

|**参数名**|**参数说明**|
|--|--|
|static_kernel_compile|布尔类型，是否开启静态Kernel编译 。False（默认值）：默认关闭。True：开启静态Kernel编译 。编译产物路径默认为当前执行脚本的同级目录，当同时启用了模型编译缓存功能但未启用force_eager功能时，编译产物路径为模型缓存文件所在目录。请确保该路径用户具有读、写操作权限。|


## 产物说明

假设产物路径为“/path/test”，目录结构如下，其中$\{timestamp\}为时间戳、$\{pid\}表示运行的进程号。

```txt
aclnn_static_shape_kernel_outputs                  // 固定的产物文件名
|—— ts${timestamp_1}_pid${pid_1}_outputs 
|    |—— ${pid_1}                                // 模型中目标算子信息文件夹
|        |—— MatMulV2_float_ND_1_2048_0.json     // js文件表示网络中的算子统计信息，包括shape和format等
|        |—— ......json
|    |—— ${pid_1}_debug                          // 模型中全量算子信息文件夹
|        |—— MatMulV2_float_ND_1_2048_0.json
|        |—— ......json
|    |—— ${pid_1}_opcompile                      // 模型中支持静态编译的算子信息导出目录
|        |—— MatMulV2_float_ND_1_2048_0.json     
|        |—— ......json
|    |—— ${pid_1}_opcompile_selected             // 模型中实际用于静态编译的算子信息导出目录；若同时启用了模型编译缓存功能，则不会生成该目录，实际使用的目录为${pid_1}_opcompile
|        |—— MatMulV2_float_ND_1_2048_0.json     
|        |—— ......json
|    |—— ${pid_1}_opcompile_gathered             // 多卡场景时模型中实际用于静态编译的算子信息导出目录，汇聚所有卡实际用于静态编译的算子信息目录，仅在rank 0生成
|        |—— MatMulV2_float_ND_1_2048_0.json     
|        |—— ......json
|    |—— static_kernel_${datetime}.run           // 编译好的静态Kernel文件；当使用多卡场景时，单次编译仅生成一份。
|    |—— ......                                  // 算子编译工具生成的编译过程文件、临时文件等
|—— ts${timestamp_2}_pid${pid_2}_outputs 
|—— ...... 
```

