# force\_recapture功能

## 功能简介

本配置项提供**强制每次执行时重新捕获aclgraph**的能力。当`force_recapture`设置为True时，aclgraph在每次执行前都会进行重新捕获，而不是复用之前捕获的图。通过对比重捕获前后执行效果，辅助问题定界。

## 使用约束

- 本功能支持的产品型号参见[使用说明](../../overview.md#使用说明)。

## 使用方法

该功能通过npugraph_ex的options配置，示例如下，仅供参考不支持直接拷贝运行，参数说明参见下表。

```python
import torch
import torch_npu
opt_model = torch.compile(model, backend="npugraph_ex", options={"force_recapture": True})
```

**表 1**  参数说明

|**参数名**|**参数说明**|
|--|--|
|force_recapture|是否强制每次执行时重新捕获aclgraph，布尔类型。<br>False（默认值）：复用已捕获的aclgraph，仅在需要时（如输入shape变化、参数变化等）才进行重新捕获。<br>True：每次执行前都强制重新捕获aclgraph。|
