# Cat算子消除功能

## 功能简介

图模式场景下，当计算图中存在 \[torch.cat\]\(<https://pytorch.org/docs/stable/generated/torch.cat.html>)（对应 \`torch.ops.aten.cat.default\`）时，TorchAir 内部会通过 Cat 算子消除 Pass 将其替换为「预分配输出张量 + slice + 原地写入」的模式，以减少内存拷贝和临时张量分配，提升执行性能。默认情况下，本功能处于开启状态。

## 使用约束

本功能仅针对 \`torch.ops.aten.cat.default\` 进行优化。

拼接维度：当前仅支持 \`dim=0\` 的 cat，其他拼接维度不会触发本优化。

输入算子约束：cat的每个输入必须来自带 \`.out\` 变体（原地变体）的算子，且不能同时作为其他cat的输入，否则该 cat 节点会被跳过。

Shape 约束：参与 cat 的各输入张量在非拼接维度上的 shape 必须一致。

若用户需要进行精度比对或问题排查，可关闭本功能以避免优化影响分析结果。

本功能支持的产品型号参见[使用说明](../../overview.md#使用说明)。

## 使用方法

该功能通过npugraph\_ex的options配置，示例如下，仅供参考不支持直接拷贝运行，参数说明参见下表。

```python
import torch
import torch_npu

torch.compile(model, backend="npugraph_ex", options={"remove_cat_ops": True}, dynamic=False, fullgraph=True)
```

**表 1**  参数说明

|**参数名**|**参数说明**|
|--|--|
|remove_cat_ops|是否开启Cat算子消除优化。<br>True（默认值）：开启优化。<br>False：关闭优化。|

开启Debug日志后，如果存在可优化的Cat节点，可以看到类似的信息：

```txt
[DEBUG] [remove_cat_ops] Found 1 cat node(s)
[DEBUG] Optimizing cat_1 (3 inputs)
[DEBUG] remove_cat_ops: Optimized 1 cat node(s)
```
