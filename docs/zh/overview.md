# 简介

## 概述

TorchAir（Torch Ascend Intermediate Representation）是昇腾Ascend Extension for PyTorch（torch\_npu）的图模式能力扩展库，提供了昇腾设备亲和的torch.compile图模式后端，实现了PyTorch网络在昇腾NPU上的图模式**推理加速**以及**性能优化**。

TorchAir在Ascend Extension for PyTorch（torch\_npu）中的位置如[图1](#fig1)所示，图中左侧为单算子执行模式（Eager），右侧为torch.compile图执行模式（Graph）。torch.compile图模式不仅继承了大部分PyTorch原生的[Dynamo特性](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamo_overview.html)（如动态shape图功能等），还在此基础上新增其他图优化和定位调试能力，例如FX图Pass优化、图内多流并行、集合通信算子入图等，手册中提到的概念请参考[常用概念](#常用概念)。

目前图执行分为两种模式：

-   **基于npugraph\_ex后端的图模式（aclgraph）**：通过设置torch.compile的backend="npugraph\_ex"开启，其采用Capture&Replay方式实现任务一次捕获多次执行。Capture阶段捕获Stream任务到Device侧，暂不执行；Replay阶段从Host侧发出执行指令，Device侧再执行已捕获的任务，从而减少Host调度开销，提升性能。

    > [!NOTE]说明
    >npugraph\_ex后端提供捕获模式（aclgraph），该模式一般通过Runtime提供的aclmdlRICaptureXxx系列接口实现，其原理和接口介绍请参考《CANN 应用开发指南 \(C&C++\)》中“运行时资源管理\>基于捕获方式构建模型运行实例”章节。

-   **基于GE的图模式（Ascend IR）**：通过设置TorchAir的CompilerConfig实例属性**mode="max-autotune"**开启，其将PyTorch的FX计算图转换为昇腾中间表示（IR，Intermediate Representation），即Ascend IR计算图，并通过GE（Graph Engine，图引擎）实现计算图的编译和执行。

**图 1**  TorchAir架构图  
![](figures/torchair_architecture.png "TorchAir架构图")<a id="fig1"></a>

## 使用说明

-   **使用场景**：当前版本的TorchAir作为**beta特性**，主要专注于**推理场景**下的模型优化。
-   **前提条件**：在使用TorchAir图模式功能之前，建议先熟悉Ascend Extension for PyTorch基础知识，详细请参见xx
-   **产品支持情况**：

    大部分功能默认支持所有产品，如有特殊情况，将在功能章节的“使用约束”中说明。

    注意npugraph\_ex后端提供的功能目前仅支持如下产品：

    -   Atlas A3 训练系列产品/Atlas A3 推理系列产品
    -   Atlas A2 训练系列产品/Atlas A2 推理系列产品

-   **整体约束**：PyTorch图模式支持单进程和多进程，每个进程**只支持使用1张NPU卡**，不支持使用多张NPU卡。
-   **npugraph\_ex后端功能约束**：
    -   当前npugraph\_ex后端提供的功能均为**试验特性**，后续版本可能存在变更，**暂不支持应用于商用产品中。**
    -   主要面向在线推理场景，暂不支持反向流程Capture成图、随机数算子Capture。
    -   npugraph\_ex与torch.cuda.CUDAGraph原生接口（参见《PyTorch 原生API支持度》中的“torch.cuda”）功能类似，约束与其保持一致（如不支持stream sync、动态控制流等），此处不再赘述。

## 兼容性说明

从torch\_npu  7.3.0之后的版本开始，**原reduce-overhead模式（aclgraph）通过config.mode配置图编译后端的方式将不再演进**，也不再推荐使用，请您切换npugraph\_ex后端以启用aclgraph模式。

若您仍需使用reduce-overhead模式（aclgraph）功能，请参考torch\_npu  7.3.0或之前版本的《[PyTorch图模式使用指南（TorchAir）](https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/torchairuseguide/torchair_00021.html)》。

## 安装

目前TorchAir暂未提供独立软件包，而是作为Ascend Extension for PyTorch的三方库，随着torch\_npu包一起发布。请直接安装torch\_npu插件，即可使用TorchAir。

torch\_npu的安装操作具体参考《[Ascend Extension for PyTorch 软件安装指南](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_description.md)》，请保证与CANN相关包的版本匹配（参见《版本说明》），否则功能可能无法正常使用。

需要注意的是：

-   当安装的torch\_npu版本为7.3.0及之后版本，均可正常使用TorchAir，对于其他torch\_npu版本请参见对应版本文档中的安装介绍。
-   为确保正常使用TorchAir功能，PyTorch建议使用2.6.0及以上版本。

## 常用概念

本节列举了手册中常用的术语和概念，以帮助您更好地理解关键特性和实现原理。

|名称|说明|
|--|--|
|Eager模式|单算子执行模式（未使用torch.compile），特点如下，单击Link获取PyTorch官网介绍。即时执行：每个计算操作在定义后立即执行，无需构建计算图。动态计算图：每次运行可能生成不同的计算图。|
|图模式|一般指使用torch.compile加速的图执行方式，特点如下：延迟执行：所有计算操作先构成一张计算图，再在会话中下发执行。静态计算图：计算图在运行前固定。|
|TorchAir图模式|PyTorch图模式（torch.compile）的一种实现，通过指定TorchAir为其backend的执行方式。|
|ATen|全称为A Tensor Library，是PyTorch张量计算的底层核心函数库，这些函数通常称为ATen算子，负责所有张量操作（如加减乘除、矩阵运算、索引等）的C++实现，单击Link获取PyTorch官网介绍。|
|FX图|Functionality Graph，PyTorch中用于表示模型计算流程的中间层数据结构。通过符号化追踪代码生成计算图，将Python代码转为中间表示（IR，Intermediate Representation），实现计算图的动态调整和优化（如量化、剪枝等），单击Link获取torch.fx详情。|
|GE|Graph Engine，图引擎。它是计算图编译和运行的控制中心，提供图优化、图编译管理以及图执行控制等功能。GE通过统一的图开发接口提供多种AI框架的支持，不同AI框架的计算图可以实现到Ascend IR图的转换，单击Link获取详情。|
|Pass|在深度学习框架（如PyTorch）和编译器（如TVM）中，Compiler Passes（编译器传递）和Partitioners（分区器）是优化图执行的关键技术，用于性能优化、硬件适配和计算图转换等，而Pass则是指在这些计算图上执行的特定变换操作。常见的Pass操作包括常量折叠、算子融合、内存优化等，单击Link获取PyTorch官网详情。FX Pass是指对计算图（torch.fx.Graph）进行遍历、分析和转换等一系列操作，类似于传统编译器中的优化步骤（如常量折叠、算子融合）。|
|In-place算子|原地算子，该类算子可直接修改输入数据，不创建新的存储空间。从而节省内存，避免复制数据的开销。|
|Out-of-place算子|非原地算子，又称“非In-place算子”，该类算子保持原始输入数据不变，会创建并返回新对象，带来额外存储开销。|
|算子Schema|在PyTorch中，算子Schema（Operator Schema）定义了算子的输入、输出、属性以及行为规范，确保算子在正向传播（Forward）和反向传播（Backward）时能正确执行。PyTorch使用Schema来注册算子，并在编译或运行时进行验证。算子Schema主要通过修改native_functions.yaml文件实现，该文件位于PyTorch源码的aten/src/ATen目录下，用于声明算子的名称、参数类型、返回值类型及设备端实现函数。|
|Ascend C|CANN编程语言，原生支持C和C++标准规范，兼具开发效率和运行性能。基于Ascend C编写的算子程序，通过编译器编译和运行时调度，运行在昇腾AI处理器上。开发的算子简称为Ascend C算子，其调用方式一般为aclnnXxx的C接口形式，具体介绍请参考《CANN Ascend C算子开发指南》。|
|OpPlugin|Ascend Extension for PyTorch（torch_npu）算子插件，为使用PyTorch框架的开发者提供便捷的NPU算子库调用能力，具体介绍参考Ascend/OpPlugin仓，而算子适配开发过程参考《PyTorch 框架特性指南》中的“基于OpPlugin算子适配开发”章节。|


