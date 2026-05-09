# 图捕获安全策略配置

## 功能简介

aclgraph图模式下，在NPU图捕获过程中，`capture_error_mode`参数用于控制对某些可能不安全操作（如分配device内存）的处理策略，该参数最终会传递给`torch_npu.npu.graph()`上下文管理器。npugraph_ex提供了`capture_error_mode`配置项，支持通过options参数灵活配置。

## 使用约束

本功能支持的产品型号参见[使用说明](../../overview.md#使用说明)。

## 使用方法

该功能通过npugraph_ex的options配置，示例如下，仅供参考不支持直接拷贝运行，参数说明参见下表。

```python
import torch
import torch_npu
opt_model = torch.compile(model, backend="npugraph_ex", options={"capture_error_mode": "global"})
```

**表 1**  参数说明

|参数名|参数说明|
|--|--|
|capture_error_mode|字符串类型，图捕获错误处理模式。<br>该参数控制NPU图捕获过程中的错误处理策略，在图捕获期间，某些操作（如分配device内存）可能是不安全的。<br>支持的模式值包括：<br>• "global"（默认值）：会在当前线程和其他线程执行这些操作时报错<br>• "thread_local"：仅在当前线程执行这些操作时报错<br>• "relaxed"：不会对这些操作报错<br>|
