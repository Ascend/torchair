# 图内标定SuperKernel范围

## 功能简介

SuperKernel是一种算子二进制融合技术，与源码融合不同，它聚焦于内核函数（Kernel）的二进制调度方案优化，在已编译的二进制代码基础上融合创建一个超级Kernel函数（简称SuperKernel），以调用子函数的方式调用多个其它内核函数，达到优化计算任务、提升性能和资源利用率的目的。

与单算子下发相比，SuperKernel技术可以优化任务调度的等待时间和调度开销，还可以利用task间隙资源进一步优化算子头开销。

实现SuperKernel的原理如下：

1.  通过SuperKernel融合策略识别可被融合的子图。
2.  将子图内的算子按SuperKernel融合规则合并为一个大Kernel，在新Kernel内通过生成一段子Kernel调用代码将子图上所有Kernel入口函数完成一次调用，并基于图的依赖完成同步插入。

TorchAir提供标定SuperKernel范围的能力，支持用户根据实际业务需求对融合范围内的算子进行标记和优化配置。

## 使用约束

-   本功能仅支持max-autotune模式，适用于静态图场景。
-   本功能支持如下产品：
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

-   需要注意的是，SuperKernel融合会按网络中算子顺序依次识别能否被融合，**若识别到不可融合的算子**，生成第一段SuperKernel，同时自动跳过该算子进行第二段SuperKernel融合。
-   目前支持SuperKernel融合的通信类算子包括AllReduce、ReduceScatter、AllGather、AlltoAll。

## 使用方法

1.  用户自行分析模型脚本中可被融合的算子。
2.  标定SuperKernel范围。

    使用如下with语句块（[super\_kernel](super_kernel.md)），语句块内算子均被融合为一个超级Kernel进行计算。

    ```
    with torchair.scope.super_kernel(scope: str, options: str = ''):
    ```

    -   scope：表示上下文算子被融合的SuperKernel名，相同的scope代表相同的范围，由用户控制。若传入None，表示该范围内的算子不进行SuperKernel融合。
    -   options：表示融合SuperKernel的编译选项，默认情况下，系统编译模式采用所有编译选项（参见[表1](#table1)）的默认值。

        同时支持用户自定义组合编译选项，配置格式形如"<option1\>=<value1\>:<option2\>=<value2\>:<option3\>=......"，多个选项时用英文冒号分割。

    **表 1**  编译选项说明  <a name="table1"></a>

    
    | 选项参数 | 说明 |
| --- | --- |
| feed-sync-all | 当子算子启动核数小于SuperKernel启动核数且使用了SyncAll全核同步指令，若出现算子执行卡住或超时，可尝试开启本功能解决此问题。<br>- 0：关闭本功能（默认值），若子算子使用SyncAll全核同步指令，用户需自行保证子算子与SuperKernel启动核数相同。<br>- 1：开启本功能，系统自动识别SuperKernel内算子是否调用SyncAll全核同步指令，同时判断子算子启动核数是否小于SuperKernel启动核数。若小于SuperKernel启动核数，会在SuperKernel内子算子的其余核中插入SyncAll指令，保证与子算子内调用SyncAll次数匹配，防止卡住超时。<br>**说明**： <br>   - SuperKernel启动核数为子算子的最大启动核数。假设SuperKernel包括算子a（启动核数为4）和算子b（启动核数为2），此时SuperKernel启动核数为4。<br>   - 启动核数可通过Profiling采集的性能数据获取，即“kernel_details.csv”文件中“Block Dim”字段，采集操作请参考[性能分析案例](性能分析案例.md)。子算子启动核数：非SuperKernel场景下开启Profiling，“Block Dim”字段表示每个算子的启动核数。SuperKernel启动核数：SuperKernel场景下开启Profiling，“Block Dim”字段表示SuperKernel的启动核数。<br>     - 子算子启动核数：非SuperKernel场景下开启Profiling，“Block Dim”字段表示每个算子的启动核数。<br>     - SuperKernel启动核数：SuperKernel场景下开启Profiling，“Block Dim”字段表示SuperKernel的启动核数。 |
| stream-fusion | 开启本功能后，可在SuperKernel内配置多流以提升算子运行效率，取值如下：<br>- 0（默认值）：表示SuperKernel内算子在单条流上执行。<br>- 1：表示SuperKernel内算子可在多条流上执行。<br>SuperKernel场景下标定的范围内算子资源共用，即不同流上的Cube和Vector算子并行执行，提高了运行效率。<br>**说明**： <br>   MicroBatch场景下，stream与核绑定且资源完全独立，建议用户使用[图内多流表达功能（Ascend IR）](图内多流表达功能（Ascend-IR）.md)（max-autotune模式）配置多流，此时不推荐同时配置stream-fusion=1。 |
| strict-scope-check | 用于检查SuperKernel融合的范围是否符合预期。对于断开的SuperKernel、不支持融合的算子可通过本功能查询：<br>- bypass：打印C++侧Warning级别日志，忽略该范围的SuperKernel生成。<br>- abort：打印C++侧Error级别日志，对该范围的SuperKernel直接报错退出。<br>**说明**： <br>  - Warning或Error级别日志信息可在plog文件中查看，搜索关键字“super_kernel_scope”即可。<br>  - plog文件一般在`$HOME/ascend/log/[run|debug]/plog`路径下，日志文件名为plog\-pid\_\*.log，\$HOME是Linux操作系统中定义的环境变量，指向当前用户的主目录路径。<br>  - 本功能暂不支持检查SuperKernel融合范围内的集合通信算子，如AllGather、AlltoAll等。 |

## 使用示例

```python
# 导入TorchAir框架
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
    print("-------------------- run no sk ----------------------------")
    swiglu_res_origin,quant_matmul_res_origin = model_no_sk(cpu_x1.npu(), cpu_x2, scale.npu(), None, None, pertoken_scale.npu(), weight_scale)
    model_sk = torch.compile(ModelSuperKernel(), backend=npu_backend, dynamic=False)
    print("-------------------- run sk ------------------------------")
    swiglu_res_sk,quant_matmul_res_sk = model_sk(cpu_x1.npu(), cpu_x2, scale.npu(), None, None, pertoken_scale.npu(), weight_scale)
    res = np.array_equal(swiglu_res_origin[0].cpu().numpy(), swiglu_res_sk[0].cpu().numpy())
    res = res and np.array_equal(swiglu_res_origin[1].cpu().numpy(), swiglu_res_sk[1].cpu().numpy())
    if res:
        print("Precision ====== Success!!!")
    else:
        print("Precision ====== Failed.")
```
