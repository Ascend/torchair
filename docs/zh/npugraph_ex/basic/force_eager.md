# force\_eager功能

## 功能简介

当模型执行出现问题，无法确定是源于npugraph\_ex的图变换操作（IR converter、Cache compile等操作）还是图执行器导致的，建议启用本配置项。

本配置项提供了**图模式执行之前以Eager模式执行FX graph**的能力，通过对比前后模型执行效果，辅助问题定界。

## 使用约束

- 当前仅npugraph\_ex提供的[FX图优化Pass配置功能](inplace_pass.md)、[静态Kernel编译功能](static_kernel_compile.md)、[模型编译缓存功能](../advanced/compile_cache.md)、[多流表达功能](../advanced/multi_stream.md)可以与本功能同时开启。
- 本功能支持的产品型号参见[使用说明](../../overview.md#使用说明)。

## 使用方法

该功能通过npugraph\_ex的options配置，示例如下，仅供参考不支持直接拷贝运行，参数说明参见下表。

```python
import torch
import torch_npu
opt_model = torch.compile(model, backend="npugraph_ex", options={"force_eager": True})
```

**表 1**  参数说明

|**参数名**|**参数说明**|
|--|--|
|force_eager|图执行前是否使用Eager模式运行，布尔类型。<br>False（默认值）：不启动Eager模式，以aclgraph图模式运行。<br>True：启动Eager模式运行。<br>支持npugraph_ex对aclgraph图增强优化功能，但是不进行aclgraph的Capture&Replay，便于进行aclgraph Runtime层问题定位。|
