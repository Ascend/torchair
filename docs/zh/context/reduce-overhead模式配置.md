# reduce-overhead模式配置

## 功能简介

> **须知：** reduce-overhead模式下的功能为试验特性，后续版本可能存在变更，暂不支持应用于商用产品中。

PyTorch原生框架默认以Eager模式运行，即单算子下发后立即执行，每个算子都需要经历如下流程：Host侧Python API-\>Host侧C++层算子下发-\>Device侧算子Kernel执行，每次Kernel执行之前都要等待Host侧下发逻辑完成。因此在单个算子计算量过小或Host性能不佳场景下，容易产生Device空闲时间，每个Kernel执行完后都需要一段时间去等待下一个Kernel下发完成。

为了优化Host调度性能，昇腾提供了NPU场景的Device调度方案，称为**aclgraph**（又称为Graph Capture，图捕获模式），将算子任务下沉到Device执行，以实现性能提升。

reduce-overhead模式是TorchAir提供的aclgraph模式开关，当用户网络存在Host侧调度问题时，建议开启此模式。

> **说明：** 
>reduce-overhead模式（aclgraph）采用Capture&Replay方式实现任务一次捕获多次执行，Capture阶段捕获Stream任务到Device侧，暂不执行；Replay阶段从Host侧发出执行指令，Device侧再执行已经捕获的任务，从而减少Host调度开销。该方案通过Runtime提供的aclmdlRICaptureXxx系列接口实现，其原理和接口介绍请参考《CANN 应用开发 \(C&C++\)》中“运行时资源管理\>基于捕获方式构建模型运行实例”章节。

## 使用约束

-   reduce-overhead模式支持如下产品：
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
-   reduce-overhead模式与max-autotune模式**不支持同时开启**。
-   reduce-overhead模式约束：
    -   主要面向在线推理场景，暂不支持反向流程capture成图、随机数算子capture。
    -   与torch.cuda.CUDAGraph原生接口（参见《PyTorch 原生API支持度》中的“torch.cuda”）功能类似，约束与其保持一致（如不支持stream sync、动态控制流等），此处不再赘述。

## 使用方法

该功能通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数介绍参见下表。

```python
import torch_npu, torchair
config = torchair.CompilerConfig()
# 设置图执行模式
config.mode = "reduce-overhead"
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

**表 1**  参数说明


| 参数名 | 说明 |
| --- | --- |
| mode | 设置图执行模式，字符串类型。<br>- max-autotune（默认值）：表示Ascend IR模式，具备一定的图融合和下沉执行能力，但要求所有算子都注册到Ascend IR计算图。<br>- reduce-overhead：表示aclgraph模式，实现了单算子Kernel下沉Device执行，暂不具备算子融合能力，不需要算子注册到Ascend IR计算图。 |

## 使用示例

端到端使用reduce-overhead模式执行的样例，请参考开源仓提供的[Deepseek V3模型NPU图模式推理样例](https://gitcode.com/Ascend/torchair/blob/master/npu_tuned_model/llm/deepseek_v3/README.md#323-%E5%9B%BE%E6%A8%A1%E5%BC%8F%E9%80%82%E9%85%8D)。

