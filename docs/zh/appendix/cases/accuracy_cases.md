# 图模式精度比对

## 精度比对流程

当PyTorch网络在昇腾NPU上进行图模式推理时，如果精度不满足预期，应该如何定位和分析问题？

首先进行问题定界，然后将复杂的整网精度问题转换为单个算子问题进行分析。

整体分析流程如下图所示，请根据实际情况逐一排查，目前核心原因集中在：
  
- 算子成Ascend IR图后导致的精度问题：
  - 先在图外将单算子单独成图编译运行，确保单算子精度正常，这是定位图模式精度问题的前提。
- Eager模式下模型存在精度问题：
  - Eager模式NPU和CPU分别运行进行精度比对。
- 原生Dynamo导致的精度问题：
  - 排查Dynamo和算子Meta推导等是否正常。
- 单个算子运行正常但局部结构成图后导致的精度问题：
  - 整网dump进行图模式与Eager模式比对，可使用精度比对工具（msit）进行比对，详见下节[精度比对工具](#精度比对工具)说明；
  - 工程师按经验分析拆除常见问题；
  - 对于GE图，可关闭算子融合规则（fusion_switch_file），排除算子融合问题，详见[算子融合配置功能（fusion_swtich_file）](../../ascend_ir/features/advanced/fusion_switch_file.md)。

**图 1**  精度问题分析流程<a name="fig1"></a>  
![](../../figures/accuracy_flowchart.png "精度问题分析流程")

## 精度比对工具

>**说明：** 
>msit工具封装了TorchAir捕获编译过程中图结构和算子数据的相关配置，与[算子data dump功能](../../ascend_ir/features/advanced/data_dump.md)相比，简化了开发者的参数设置，并在此基础上提供了精度比对等高级功能。

MindStudio（msit工具包）推理工具链为开发者提供了一站式推理开发工具，包括模型压缩、推理数据dump、自动精度比对、性能调优等能力。

精度比对能力一般借助大模型推理精度工具（Large Language Model Debug Tool）实现，其软件包安装参考[msit大模型推理精度工具](https://gitcode.com/Ascend/msit/tree/master/msit/docs/llm)，在“简介\>工具列表”中获取**精度自动比对功能**介绍。

关键命令如下，此处仅为示例，请以开源仓的说明为准，全量参数介绍参见[精度比对命令完整参数](https://gitcode.com/Ascend/msit/blob/master/msit/docs/llm/%E5%B7%A5%E5%85%B7-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%B2%BE%E5%BA%A6%E6%AF%94%E5%AF%B9.md#21-%E5%91%BD%E4%BB%A4%E8%A1%8C%E6%96%B9%E5%BC%8F)。

```bash
msit llm compare -gp ${golden_data} -mp ${target_data} -o ${compare_result_dir}
```

**表 1**  关键命令参数

|参数|说明|
|--|--|
|-gp，--golden-path|指定标杆数据所在路径，支持单个数据文件路径或文件夹。|
|-mp，--my-path|指定待比对的数据所在路径，即有精度问题的数据。支持单个数据文件路径或文件夹。|
|-o，--output|指定比对结果保存路径。|

借助msit工具进行图模式与Eager模式下模型精度差异比对的原理如下图所示，具体过程可参考[精度比对案例](#精度比对案例)。

![](../../figures/251020112652416.png)

## 精度比对案例

假设有一个PyTorch模型，在图模式下出现了精度异常。为确定问题来源，借助msit工具比对图模式与Eager模式下模型精度差异，请参见[精度比对工具](#精度比对工具)，具体操作步骤如下：

>**说明：** 
>关于TorchAir图模式下FX图和GE图数据dump、compare等详细介绍，请参考[msit大模型推理精度工具](https://gitcode.com/Ascend/msit/tree/master/msit/docs/llm)，在“简介\>场景列表”中获取TorchAir场景-整网算子精度比对介绍。

1. 环境准备，参考[安装](../../overview.md#安装)完成torch\_npu安装和依赖的软件安装。
2. 安装msit工具包里llm组件（大模型推理精度工具）。
    1. 首先安装msit工具。

        这里以源码安装为例，详细安装过程和命令介绍请参考[msit工具安装](https://gitcode.com/Ascend/msit/tree/master/msit/docs/install)。

        ```bash
        git clone https://gitcode.com/ascend/msit.git
        cd msit/msit
        pip install .
        ```

    2. 下载llm组件包。

        使用msit download下载命令，--dest表示存放的指定目录。

        ```bash
        msit download llm --dest ${llm_dir}
        ```

    3. 安装llm组件包。

        使用msit install安装命令，--find-links表示待安装的文件目录。

        ```bash
        msit install llm --find-links ${llm_dir}
        ```

    4. 检查是否安装成功。

        使用msit check命令检查安装结果，日志会提示“msit-llm”安装成功。

        ```bash
        msit check all
        ```

3. 获取图模式下Ascend IR图dump数据。

    以如下脚本为例，在torch.compile入图处使用msit提供的[get\_ge\_dump\_config接口](https://gitcode.com/Ascend/msit/blob/master/msit/docs/llm/TorchAir%E5%9C%BA%E6%99%AF-%E6%95%B4%E7%BD%91%E7%AE%97%E5%AD%90%E7%B2%BE%E5%BA%A6%E6%AF%94%E5%AF%B9.md#11-ge-%E8%9E%8D%E5%90%88%E6%A8%A1%E5%BC%8F-dump-%E6%95%B0%E6%8D%AE)获取Ascend IR图数据。

    ```python
    # 导包
    import torch, torch_npu, torchair 
    from msit_llm.dump import torchair_dump  
    
    # 定义模型Model
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, y):
            ......
    model = Model().npu()
    # 配置获取Ascend IR图dump的config
    config = torchair_dump.get_ge_dump_config(dump_path=${ge_dump_path})  
    ......
    # Graph模式下torch.compile 
    npu_backend = torchair.get_npu_backend(compiler_config=config) 
    model = torch.compile(model, backend=npu_backend, dynamic=True) 
    ......
    ```

    配置后执行推理脚本，会在dump\_path指定目录下生成dump数据，目录样式如下：

    ```text
    |--${ge_dump_path}
       |--dump_${timestamp}
         |--${op_dump_timestamp}   // 图模式下dump的算子输入/输出信息文件夹，以时间戳命名
         |--dynamo_original_graph_${graph_id}_rank_${rank_id}_pid_${pid}_ts_${timestamp}.txt    // 图模式下dump的原始图结构
         |--dynamo_optimized_graph_${graph_id}_rank_${rank_id}_pid_${pid}_ts_${timestamp}.txt     // 图模式下dump的优化后图结构
    ```

    dump的数据包括两部分，一部分是“图模式下dump的算子输入/输出信息”，文件介绍可参考[算子data dump功能](../../ascend_ir/features/advanced/data_dump.md)中“产物说明”；另一部分是“图模式下dump的图结构信息”，文件介绍可参考[图结构dump功能](../../ascend_ir/features/basic/graph_dump.md)中的产物说明”。

4. 获取Eager模式下FX图dump数据。

    以如下脚本为例，在torch.compile入图处使用msit提供的[get\_fx\_dump\_config接口](https://gitcode.com/Ascend/msit/blob/master/msit/docs/llm/TorchAir%E5%9C%BA%E6%99%AF-%E6%95%B4%E7%BD%91%E7%AE%97%E5%AD%90%E7%B2%BE%E5%BA%A6%E6%AF%94%E5%AF%B9.md#12-fx-%E6%A8%A1%E5%BC%8F-dump-%E6%95%B0%E6%8D%AE)获取FX图数据。

    ```python
    import torch, torch_npu, torchair 
    from msit_llm.dump import torchair_dump  
    
    # 定义模型Model
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, y):
            ......
    model = Model().npu()
    # 配置获取FX图dump的config
    config = torchair_dump.get_fx_dump_config()  
    ......
    # Eager模式下torch.compile
    npu_backend = torchair.get_npu_backend(compiler_config=config) 
    model = torch.compile(model, backend=npu_backend, dynamic=True) 
    ......
    ```

    配置后执行推理脚本，一般会在当前路径下data\_dump/$\{token\_id\}/gm\_$\{time stamp\}\_dump（老版本中路径可能为gm\_$\{timestamp\}\_dump）目录生成dump数据，其中$\{token\_id\}从1开始，相对于GE模式是从0开始的，比对时会将FX模式的$\{token\_id\}减1。产物是npy格式，文件名和内容介绍可参考[算子data dump功能（Eager模式）](../../ascend_ir/features/basic/data_dump_eager.md)中“产物说明”。

5. 通过llm组件提供的精度比对能力，比对两种模式下的模型精度。

    使用如下命令，请确保当前用户已拥有目标文件的读、写权限。

    ```bash
    msit llm compare --my-path ${ge_dump_path}/dump_${timestamp} --golden-path ${fx_dump_path}
    ```

    - $\{ge\_dump\_path\}：图模式下Ascend IR图dump数据路径，即get\_ge\_dump\_config接口dump\_path参数指定路径下$\{ge\_dump\_path\}/dump\_$\{timestamp\}目录。
    - $\{timestamp\}：图模式下dump对应的时间戳。
    - $\{fx\_dump\_path\}：Eager模式下FX dump数据路径，即get\_fx\_dump\_config接口默认路径下data\_dump目录。

    执行命令后，会出现类似的回显信息，精度比对结果文件中的参数说明请参见[精度比对结果参数说明](https://gitcode.com/Ascend/msit/blob/master/msit/docs/llm/%E7%B2%BE%E5%BA%A6%E6%AF%94%E5%AF%B9%E7%BB%93%E6%9E%9C%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E.md)，以便做进一步分析。

    ```bash
    msit_llm_logger - INFO - Comparing GE with FX
    msit_llm_logger - INFO - All token ids in my_dump_data: dict_keys([0])
    ......
    msit_llm_logger - INFO - Saved comparing results: ./msit_cmp_report_${timestamp}.csv
    ```
