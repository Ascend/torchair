# 重捕获次数限制功能

## 功能简介

npugraph\_ex后端使能（aclgraph）模式下，由于aclgraph本身不支持动态shape，因此执行过程中可能会因为输入shape的变化而多次重新捕获aclgraph。

重新捕获的过程会带来额外的性能开销和Device侧资源消耗，尤其是频繁的重捕获，可能导致资源不足。

本功能提供了一个配置项，用于控制单个FX图或子图中不同shape的重捕获次数阈值，针对每个FX图或子图单独生效。当某个FX图或子图重捕获超过设置的阈值时，该图或子图后续的所有执行将全部回退到Eager模式。

## 使用约束

本功能支持的产品型号参见[使用说明](../../overview.md#使用说明)。

## 使用方法

```python
import torch
import torch_npu

opt_model = torch.compile(model, backend="npugraph_ex", options={"capture_limit": 64}, fullgraph=True, dynamic=False)
```

该功能配置示例如下，仅供参考不支持直接拷贝运行。

**表 1**  参数说明

|参数名|说明|
|--|--|
|capture_limit|int类型，最小值1，最大值为9223372036854775807。<br>64（默认值）：默认允许的重捕获次数为64。<br>注意：当重捕获时，NPU设备上stream、event、memory等资源超限时，会因为资源不足退出流程，可能无法触发到最大值。|


