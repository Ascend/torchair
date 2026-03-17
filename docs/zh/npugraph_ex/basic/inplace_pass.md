# FX图优化Pass配置功能

## 功能简介

aclgraph图模式下，可基于torch.compile生成的ATen IR表示FX Graph或GraphModule，通过对图中的ATen IR变换和分析，可在不修改原始模型代码的情况下对模型进行灵活修改，例如应用各种图优化Pass操作（算子融合、精度转换、量化等）。

TorchAir提供了一些FX Pass配置项，可以将变换后的ATen IR下沉到aclgraph执行器上，提升算子的执行效率。

## 使用约束

本功能支持的产品型号参见[使用说明](../../overview.md#使用说明)。

## 使用方法

该功能通过npugraph\_ex的options配置，示例如下，仅供参考不支持直接拷贝运行，参数说明参见下表。

```python
import torch
import torch_npu
opt_model = torch.compile(model, backend="npugraph_ex", options={"inplace_pass": True, "input_inplace_pass": True})
```

**表 1**  参数说明

|参数名|参数说明|
|--|--|
|inplace_pass|布尔类型，是否开启Pass。<br>该Pass针对模型的中间节点，将节点中包含的Out-of-place算子（非原地算子）替换为In-place算子（原地算子），以减少计算过程中的内存搬运，从而提升性能。<br>True（默认值）：默认开启此Pass。<br>False：关闭此Pass。<br>多原地算子（修改了多个输入的算子）中，当前仅支持torch_npu.npu_kv_rmsnorm_rope_cache_v2（试验接口，后续版本可能存在变更，暂不支持应用于商用产品中），torch_npu.npu_mla_prolog_v3，torch_npu.npu_add_rms_norm_v2三个算子开启本pass。|
|input_inplace_pass|布尔类型，是否开启Pass。<br>该Pass针对模型的原始输入参数，若包含原地操作算子（如KV Cache），Dynamo的Functionalize流程会将原地算子替换为“Out-of-place算子+copy_算子”，本Pass是该操作的逆向过程，将“Out-of-place算子+copy_算子”替换为In-place算子，以减少计算过程中的内存搬运，从而提升性能。<br>True（默认值）：默认开启此Pass。<br>False：关闭此Pass。<br>多原地算子中，当前仅支持torch_npu.npu_kv_rmsnorm_rope_cache_v2（试验接口，后续版本可能存在变更，暂不支持应用于商用产品中），torch_npu.npu_mla_prolog_v3，torch_npu.npu_add_rms_norm_v2三个算子开启本pass。|


