# 集合通信入图

## 功能简介

集合通信入图能够避免断图，并拥有更大的成图范围，从而获得更大的资源调度与融合收益，同时在整图层面实现通信与计算并行优化。

原生PyTorch社区对集合通信算子入图的支持度尚不完善，功能正在持续增强中。针对这一情况，npugraph\_ex实现了集合通信算子的Ascend Converter，在调用torch.compile时默认已支持集合通信算子入图，具体入图方法参见[使用方法](#使用方法)。

目前支持入图的集合通信API如下表所示，请根据实际业务需求按需调用。注意，集合通信算子入图的前提是PyTorch脚本中所有算子均能正常以Eager模式运行。

**表 1**  集合通信API入图支持情况

|PyTorch集合通信API|支持情况|说明|
|--|--|--|
|torch.distributed.all_gather|√|torch_npu接口详细介绍请参考《Ascend Extension for PyTorch 自定义API参考》，其余接口均为PyTorch原生接口。|
|torch.distributed.all_gather_into_tensor|√|
|torch.distributed.all_reduce|√|
|torch.distributed.all_to_all|√|
|torch.distributed.all_to_all_single|√|
|torch.distributed.broadcast|√|
|torch.distributed.reduce_scatter_tensor|√|
|torch_npu.distributed.all_gather_into_tensor_uneven|√|
|torch_npu.distributed.reduce_scatter_tensor_uneven|√|
|torch.distributed.send|√|
|torch.distributed.recv|√|

> [!NOTE]说明
>
>- torch.distributed.send和torch.distributed.recv需要配套使用，且dynamic=True的场景，不同的shape会对应不同的FX graph。
>- max-autotune模式下，torch.distributed.send和torch.distributed.recv不传入group参数时需要有默认通信组（所有节点都有send/recv或者提前建好全局默认通信域），传入group参数时应当只包含参与通信的节点；当图中存在多个torch.distributed.send、torch.distributed.recv时，需要设置图遍历顺序为StableRDFS（稳定拓扑序策略）。

## 使用约束

本功能支持的产品型号参见[使用说明](../../overview.md#使用说明)。

## 使用方法

无需修改PyTorch脚本，直接调用torch.compile，NPU图编译后端npugraph\_ex默认集成了集合通信算子入图能力。

```python
import torch
import torch_npu

# 多卡模型调用compile，后端提供集合通信入图能力
model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=False)
```
