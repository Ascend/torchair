# Tiling调度优化功能

## 功能简介

在静态图场景下，可以通过整图下沉优化调度性能。将完整的计算图一次性下发至Device侧，后续执行则无需Host参与，由Device自主完成计算，从而减少Host-Device交互开销，提升执行效率。部分算子的Tiling计算依赖运行时输入的具体数值（Tiling值依赖），需在执行时动态计算Tiling参数。针对该场景，可采用**Tiling下沉**优化方案：将Tiling计算下沉至Device侧的AI CPU上执行，从而实现计算全程在Device侧高效完成。

> **说明：** 
>Tiling计算描述了NPU上算子输入/输出数据切分、分块计算、多核并行等逻辑，以满足片上存储限制和计算pipeline的需求，充分发挥硬件性能。

## 使用约束

-   本功能仅支持max-autotune模式，模型为静态Shape。
-   当前仅融合算子（矢量计算和矩阵计算融合）支持Tiling下沉，例如FusedInferAttentionScore、IncreFlashAttention。
-   基于新版本CANN包（支持Tiling下沉特性）编译生成的Tiling下沉算子，不兼容旧版CANN（不支持Tiling下沉特性）运行环境。
-   本功能支持如下产品：
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>

## 使用方法

该功能通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数介绍参见下表。

```python
import torch_npu
import torchair 
config = torchair.CompilerConfig()
# Tiling调度优化配置
config.experimental_config.tiling_schedule_optimize = True
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

**表 1**  参数说明


| 参数名 | 说明 |
| --- | --- |
| tiling_schedule_optimize | 是否开启Tiling计算调度优化。<br>- False（默认值）：不开启。<br>- True：开启。 |

## 使用示例

本文档仅提供Tiling下沉开关配置介绍，端到端算子Tiling下沉过程可以访问Ascend samples仓，获取图模式下自定义算子[AddCustomTilingSink下沉样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/2_features/17_tiling_sink/AddCustomTilingSink)，请仔细阅读README.md。

