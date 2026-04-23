# 背景介绍

## 编译流程简介

介绍动/静态图概念之前，请参考[简介](../../../overview.md)了解TorchAir架构图，max-autotune模式主要是将PyTorch的FX计算图转换为Ascend IR计算图，并通过（Graph Engine，图引擎）实现图编译和执行。

模型的编译流程如下图所示，脚本执行经过Dynamo编译、TorchAir图优化和GE图编译，最终编译生成GE build图。Dynamo编译后的图称为FX原图，TorchAir优化后的图称为GE原图，GE图编译之后的图称为GE build图。

**图 1**  模型编译流程  
![](../../../figures/model_compile.png "模型编译流程")

无论是FX图还是GE图，均区分动态图和静态图。本文将深入介绍**Dynamo和GE中的动/静态图概念**，并介绍TorchAir和GE组件的衔接。

此外，本文提供了动、静态图场景下常见的**Tiling调度问题**，以依赖FlashAttention（FA）的算子在GE中下沉调度为例，分析其下沉条件及下沉调度问题的基本定位思路。

# TorchAir与GE交互流程

编译和执行阶段TorchAir和GE的衔接时序如下图：

![](../../../figures/zh-cn_image_0000002512422333.png)

1. Converter前FX图优化：如特殊inplace\_pattern优化、sym\_input优化、view\_to\_reshape优化。
2. Converter：实现Aten IR转换为Ascend IR。
3. Converter后FX图优化：如死节点消除、符号输入转换为ge.Data等。
4. 优化后的图，经过反序列化加载得到GE Model对象。
5. 首次执行，触发向GE Session添加graph的动作。
6. 首次执行，触发向GE Session的graph的编译动作。
7. TorchAir根据编译结果，生成对应的Executor执行器。
8. 调用GE图执行接口，如ExecuteGraphWithStreamAsync。
