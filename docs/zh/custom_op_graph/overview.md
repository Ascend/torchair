# 概述

## 基本介绍

本章介绍自定义PyTorch算子如何与TorchAir图模式配合工作（也叫**自定义算子入图**）。阅读本节，您将了解自定义PyTorch算子与TorchAir配合的完整流程，以及流程涉及的交付件及其作用。

TorchAir是昇腾为PyTorch图模式（torch.compile）实现的NPU后端，提供两种工作模式：

-   npugraph\_ex后端使能（aclgraph）模式，提供模型下沉调度、多Stream并行、图间内存复用等能力。该模式与Torch原生图模式所需的交付件完全一致。
-   GE图模式（mode=max-autotune），又称为Ascend IR模式，在aclgraph能力基础上额外提供SuperKernel等JIT编译能力，进一步提升执行性能。该模式需要实现Ascend Converter交付件，完成PyTorch算子转换为Ascend IR。 

实现一个能与TorchAir图模式配合工作的PyTorch算子**最多包含如下步骤**（以实现In-place类算子为例）。

1.  PyTorch Eager模式调用交付件，确定目标PyTorch算子原型，并完成Schema定义。
2.  PyTorch Eager模式调用交付件，基于Ascend C完成目标PyTorch算子的NPU实现。
3.  PyTorch Eager模式调用交付件，基于OpPlugin完成Ascend C算子Eager模式适配，实现Eager模式下调用运行在NPU上的PyTorch算子。
4.  PyTorch原生入图操作，完成Meta符号化推导。完成后可使用TorchAir npugraph\_ex后端使能（aclgraph）模式以及aot\_eager等原生图模式后端，并获取基于aclgraph的下沉调度收益。
5.  （可选）PyTorch原生入图操作，**仅当算子为In-place类算子**才需要额外完成本步骤。
6.  （可选）Ascend IR入图操作，如果您希望使用[SuperKernel](../ascend_ir/features/advanced/super_kernel_scope.md)、[分核执行](../ascend_ir/features/advanced/limit_cores.md)等max-autotune模式提供的高阶能力，才需要额外完成本步骤。

**使用说明**
-   本章中提到的Eager模式、算子Schema、Ascend C、OpPlugin、In-place算子等术语请先参考[常用概念](../overview.md#常用概念)进行了解，避免概念混淆。
-   对于图中的“**Meta推导函数**”，PyTorch原生要求所有能与torch.compile配合工作的算子需要实现Meta推导函数，又称为“符号化推导”。Meta函数表示了PyTorch算子输出与输入shape、dtype以及内存的关系，它是PyTorch入图的前提条件，其详细介绍请参考PyTorch官网<u>[符号化手册](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.fh8zzonyw8ng)</u>。
-   对于图中的“**函数化**”，请您先了解In-place与非In-place操作：
    
    - 计算时会修改输入的算子称为In-place算子（或原地算子、Ref类算子等），例如torch.ops.aten.add\_。

    - 与torch.ops.aten.add\_对应的非In-place算子为torch.ops.aten.add，其结果写入输出而非直接修改输入。

    “函数化转换”可以简单理解为将In-place算子替换为非In-place算子的过程，例如将torch.ops.aten.add\_替换为torch.ops.aten.add。PyTorch图模式基于函数化后的FX图工作，因此In-place类算子与PyTorch图模式配合工作时，需要实现函数化转换，实现将图上的In-place算子替换为非In-place算子。
-   注意In-place与非In-place算子的交付件略有不同：非In-place算子本身是函数化的，无需实现函数化转换。**In-place类算子需要实现函数化**，TorchAir已经支持了PyTorch社区的**自动化函数化**能力。

**图 1**  算子入图流程  
![](../figures/op_in_graph_flowchart.png "算子入图流程")

## 场景说明

对于不同的入图场景，所需实现的步骤不同。

-   **场景1**：当您未完成自定义算子开发，希望开发Eager模式和图模式下都能工作的PyTorch算子。

    **您需要完成图中步骤1至步骤6**，“可选”步骤请按需实现，详细的操作参考[算子开发和适配TorchAir图模式](op_adapt_torchair.md)。

-   **场景2**：当您已完成自定义算子开发且能在Eager模式下正常工作，希望完成后续的PyTorch图模式适配。

    **您需要完成图中步骤4至步骤6**，“可选”步骤请按需实现，详细的操作参考[算子插件化适配TorchAir图模式](op_plugin_adapt_torchair.md)。

    该入图方式称为“插件化适配”，过程中无需编译、安装torch\_npu，并且采用全Python实现，您可以在任意py文件中实现上述交付件，并在模型执行前加载即可。方便进行算子调试或者将自定义算子模块作为插件使用。

## 样例说明

针对不同的入图场景、不同的算子类型（In-place/非In-place算子），本章提供了对应的开发示例（罗列关键步骤，请根据实际业务场景调整示例代码）：

-   **场景1**：当您未完成自定义算子开发，希望开发Eager模式和图模式下都能工作的PyTorch算子。
    -   [非In-place算子开发和入图样例](non_in_place_op_cases.md)
    -   [In-place算子开发和入图样例](in_place_op_cases.md)

-   **场景2**：当您已完成自定义算子开发且能在Eager模式下正常工作，希望完成后续的PyTorch图模式适配。
    -   [非In-place算子插件化入图样例](./op_plugin_adapt_torchair.md#非In-place算子插件化入图样例)
    -   [In-place算子插件化入图样例](./op_plugin_adapt_torchair.md#In-place算子插件化入图样例)

## 环境准备

### 软件安装

实现自定义PyTorch算子的图模式需要安装如下软件：

-   PyTorch：本章所有样例建议下载2.6.0版本。
-   Ascend Extension for PyTorch（torch\_npu）：注意与CANN等软件配套关系。
-   CANN软件：注意与torch\_npu等软件配套关系。
-   固件/驱动等：注意与CANN、torch\_npu等软件配套关系。

安装前请注意**软件版本配套关系**（《版本说明》），安装指导参考《PyTorch 框架特性指南》中的“基于OpPlugin算子适配开发”章节，完成编译及运行依赖项的安装，保证可以正常编译、安装及执行torch\_npu。

基于torch\_npu自定义算子接入流程开发，与torch\_npu源码一起编译打包，该方式便于您与他人共享自定义算子。

### torch\_npu源码下载

下载[torch\_npu源码](https://gitcode.com/Ascend/pytorch)时，请注意与当前运行环境的PyTorch版本匹配。以PyTorch 2.6.0版本为例，下载命令如下：

```bash
git clone https://gitcode.com/Ascend/pytorch.git -b v2.6.0 --recursive
cd pytorch
```


