# 图内标定SuperKernel范围

## 功能简介

SuperKernel是一种算子二进制融合技术，与源码融合不同，它聚焦于内核函数（Kernel）的二进制调度方案优化，在已编译的二进制代码基础上融合创建一个超级Kernel函数（简称SuperKernel），以调用子函数的方式调用多个其它内核函数，达到优化计算任务、提升性能和资源利用率的目的。

与单算子下发相比，SuperKernel技术可以优化任务调度的等待时间和调度开销，还可以利用task间隙资源进一步优化算子头开销。

实现SuperKernel的原理如下：

1.  通过SuperKernel融合策略识别可被融合的子图。
2.  将子图内的算子按SuperKernel融合规则合并为一个大Kernel，在新Kernel内通过生成一段子Kernel调用代码将子图上所有Kernel入口函数完成一次调用，并基于图的依赖完成同步插入。

TorchAir提供标定SuperKernel范围的能力，支持用户根据实际业务需求对融合范围内的算子进行标记和优化配置。

## 使用约束

-   本功能仅适用于GE图模式场景，并且需要为静态图场景。
-   本功能支持如下产品：
    -   Atlas A3 训练系列产品/Atlas A3 推理系列产品
    -   Atlas A2 训练系列产品/Atlas A2 推理系列产品

-   需要注意的是，SuperKernel融合会按网络中算子顺序依次识别能否被融合，**若识别到不可融合的算子**，生成第一段SuperKernel，同时自动跳过该算子进行第二段SuperKernel融合。
-   目前支持SuperKernel融合的通信类算子包括AllReduce、ReduceScatter、AllGather、AlltoAll。

## 使用方法

1.  用户自行分析模型脚本中可被融合的算子。
2.  标定SuperKernel范围。

    使用如下with语句块（[super\_kernel](../../api/scope/super_kernel.md)），语句块内算子均被融合为一个超级Kernel进行计算。

    ```python
    with torchair.scope.super_kernel(scope: str, options: str = ''):
    ```

    -   scope：表示上下文算子被融合的SuperKernel名，相同的scope代表相同的范围，由用户控制。若传入None，表示该范围内的算子不进行SuperKernel融合。
    -   options：表示融合SuperKernel的编译选项，默认情况下，系统编译模式采用所有编译选项（参见[表1](#tab1)）的默认值。

        同时支持用户自定义组合编译选项，配置格式形如"<option1\>=<value1\>:<option2\>=<value2\>:<option3\>=......"，多个选项时用英文冒号分割。

    **表 1**  编译选项说明<a id="tab1"></a>

    |选项参数|说明|
    |--|--|
    |feed-sync-all|当子算子启动核数小于SuperKernel启动核数且使用了SyncAll全核同步指令，若出现算子执行卡住或超时，可尝试配置本选项解决此问题。<br>0：关闭本功能（默认值），若子算子使用SyncAll全核同步指令，用户需自行保证子算子与SuperKernel启动核数相同。<br>1：开启本功能，系统自动识别SuperKernel内算子是否调用SyncAll全核同步指令，同时判断子算子启动核数是否小于SuperKernel启动核数。若小于SuperKernel启动核数，会在SuperKernel内子算子的其余核中插入SyncAll指令，保证与子算子内调用SyncAll次数匹配，防止卡住超时。<br>SuperKernel启动核数为子算子的最大启动核数。假设SuperKernel包括算子a（启动核数为4）和算子b（启动核数为2），此时SuperKernel启动核数为4。<br>启动核数可通过Profiling采集的性能数据获取，即“kernel_details.csv”文件中“Block Dim”字段，采集操作请参考[性能分析案例](../../../appendix/cases/perfermance_cases.md#性能分析案例)。<br>子算子启动核数：非SuperKernel场景下开启Profiling，“Block Dim”字段表示每个算子的启动核数。<br>SuperKernel启动核数：SuperKernel场景下开启Profiling，“Block Dim”字段表示SuperKernel的启动核数。|
    |stream-fusion|配置本选项后，可在SuperKernel内配置多流，从而提升算子运行效率，选项取值如下：<br>0（默认值）：表示SuperKernel内算子在单条流上执行。<br>1：表示SuperKernel内算子可在多条流上执行。<br>SuperKernel场景下标定的范围内算子资源共用，当标定范围内存在纯Cube算子与纯Vector算子分别运行在不同stream上的并行场景时，配置stream-fusion=1后，在SuperKernel内的Cube算子和Vector算子可以并行执行，通常能获得较好的性能收益。对于其他并行场景，SuperKernel内部算子可能以串行方式执行，因此不一定能获得性能收益。<br>如需在图中显式配置多流，建议使用图内[多流表达功能](multi_stream.md)；该功能支持在脚本中手动指定算子的执行stream，将可并行的算子分发到不同stream上。|
    |strict-scope-check|本选项用于检查SuperKernel融合的范围是否符合预期。对于断开的SuperKernel、不支持融合的算子可通过本功能查询：<br>bypass：打印C++侧Warning级别日志，忽略该范围的SuperKernel生成。<br>abort：打印C++侧Error级别日志，对该范围的SuperKernel直接报错退出。<br>Warning或Error级别日志信息可在plog文件（文件名为plog-*pid_**.log）中查看，搜索关键字“super_kernel_scope”即可。<br>本功能暂不支持检查SuperKernel融合范围内的集合通信算子，如AllGather、AlltoAll等。|
    |dcci-before-kernel-start|通过本选项指定的算子，其内部调用GlobalTensor的GetValue/SetValue时不会自动插入缓存刷新指令，**而在SuperKernel调用该算子前**会插入DataCacheCleanAndInvalid指令，刷新整个DCache（数据缓存），保证该算子内的数据缓存不受前序算子的影响。<br>配置格式形如：dcci-before-kernel-start=<op1_type>,<op2_type><br>本选项在保证算子本身cache一致性的前提下可提升模型性能，原理可参考《CANN Ascend C算子开发指南》中“编程指南>附录>算子入图（GE图）开发>SuperKernel开发”章节。<br>若本选项指定的算子为支持Tiling下沉的算子或通信类算子，功能将不生效。<br>DataCacheCleanAndInvalid接口介绍参见《CANN Ascend C算子开发接口》中的“基础API>缓存处理>DataCacheCleanAndInvalid”章节。<br>本选项指定的算子op_type可通过Profiling查看，例如：dcci-before-kernel-start=GroupedMatmul,MoeGatingTopK。|
    |dcci-after-kernel-end|通过本选项指定的算子，其内部调用GlobalTensor的GetValue/SetValue时不会自动插入缓存刷新指令，**而在SuperKernel调用该算子后**，会插入DataCacheCleanAndInvalid指令，刷新整个DCache（数据缓存），保证该算子内的数据缓存不会影响后续的算子。<br>配置格式形如：dcci-after-kernel-end=<op1_type>,<op2_type>。<br>其他要求与dcci-before-kernel-start一样。|
    |dcci-disable-on-kernel|通过本选项指定的算子，其内部调用GlobalTensor的GetValue/SetValue时不会自动插入缓存刷新指令，**而在SuperKernel调用该算子前后**，不会插入任何DataCacheCleanAndInvalid指令。<br>配置格式形如：dcci-disable-on-kernel=<op1_type>,<op2_type>。<br>其他要求与dcci-before-kernel-start一样。|
    |debug-aic-num|本选项用于指定Scope内SuperKernel最终启动的aic核数（Cube Core），要求正整数，配置格式形如：debug-aic-num=10。<br>本选项通常与debug-aiv-num一同使用，对应不同的kernel type，组合效果如下：<br>debug-aic-num=12，SK按照MIX_AIC_1_0启动12核。<br>debug-aic-num=12:debug-aiv-num=0，SK按照MIX_AIC_1_0启动12核。<br>debug-aiv-num=12，SK按照MIX_AIV_1_0启动12核。<br>debug-aiv-num=12:debug-aic-num=0，SK按照MIX_AIV_1_0启动12核。<br>debug-aic-num=12:debug-aiv-num=12，SK按照MIX_AIC_1_1启动12核。<br>debug-aic-num=12:debug-aiv-num=24，SK按照MIX_AIC_1_2启动12核。<br>debug-aic-num与debug-aiv-num的比例仅支持1:0、0:1、1:1、1:2。<br>debug-aic-num与debug-aiv-num的数值不可超过当前硬件最大核数，若SuperKernel Scope内算子使用[limit_core_num](../../api/scope/limit_core_num.md)控制核数，则不可超过其设置的最大核数。<br>debug-aic-num与debug-aiv-num的数值不可超过SuperKernel原本应启动的核数。|
    |debug-aiv-num|本选项用于指定Scope内SuperKernel最终启动的aiv核数（Vector Core），要求正整数，配置格式形如：debug-aiv-num=10。其他要求与debug-aic-num一样。|


## 使用示例

```python
import torch
import numpy as np
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig

if __name__ == "__main__":   
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    
    # 定义模型model
    class ModelOrigin(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, scale, offset, bias, pertoken_scale, weight_scale):
            quant_matmul_res_origin = torch_npu.npu_quant_matmul(x1, x2, scale, offset=offset, bias=bias, pertoken_scale=pertoken_scale, output_dtype=torch.bfloat16)
            swiglu_res_origin = torch_npu.npu_dequant_swiglu_quant(quant_matmul_res_origin, weight_scale=weight_scale) 
            return swiglu_res_origin, quant_matmul_res_origin
    class ModelSuperKernel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, scale, offset, bias, pertoken_scale, weight_scale):
            # 将 npu_quant_matmul和npu_dequant_swiglu_quant融合为superKernel，标记为sp1
            with torchair.scope.super_kernel("sp1",""):
                quant_matmul_res_origin = torch_npu.npu_quant_matmul(x1, x2, scale, offset=offset, bias=bias, pertoken_scale=pertoken_scale, output_dtype=torch.bfloat16)
                swiglu_res_origin = torch_npu.npu_dequant_swiglu_quant(quant_matmul_res_origin, weight_scale=weight_scale) 
                return swiglu_res_origin, quant_matmul_res_origin
    
    m = 864
    k = 7168
    n = 4096
    bias_flag = False
    cpu_x1 = torch.randint(-10, 10, (m, k), dtype=torch.int8)
    cpu_x2 = torch.randint(-10, 10, (n, k), dtype=torch.int8)
    cpu_x2 = torch_npu.npu_format_cast(cpu_x2.npu().transpose(1,0).contiguous(), 29)
    scale = torch.randn((n,), dtype=torch.float32)
    # print("scale:", scale)
    pertoken_scale = torch.randn((m,), dtype=torch.float32)
    # print("pertoken_scale:", pertoken_scale)
    bias = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
    weight_scale = torch.randn((m, n), dtype=torch.float32).npu()
    # 使用图模式后端编译模型
    model_no_sk = torch.compile(ModelOrigin(), backend=npu_backend, dynamic=False)
    print("-------------------- run no sk -----------------------------------")
    swiglu_res_origin,quant_matmul_res_origin = model_no_sk(cpu_x1.npu(), cpu_x2, scale.npu(), None, None, pertoken_scale.npu(), weight_scale)
    model_sk = torch.compile(ModelSuperKernel(), backend=npu_backend, dynamic=False)
    print("-------------------- run sk -----------------------------------")
    swiglu_res_sk,quant_matmul_res_sk = model_sk(cpu_x1.npu(), cpu_x2, scale.npu(), None, None, pertoken_scale.npu(), weight_scale)
    res = np.array_equal(swiglu_res_origin[0].cpu().numpy(), swiglu_res_sk[0].cpu().numpy())
    res = res and np.array_equal(swiglu_res_origin[1].cpu().numpy(), swiglu_res_sk[1].cpu().numpy())
    if res:
        print("Precision ====== Success!!!")
    else:
        print("Precision ====== Failed.")
```

