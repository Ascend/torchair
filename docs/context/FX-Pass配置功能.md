# FX Pass配置功能

## 功能简介

>**须知：** reduce-overhead模式下的功能为试验特性，后续版本可能存在变更，暂不支持应用于商用产品中。

reduce-overhead执行模式（aclgraph）下，可基于torch.compile生成的[ATen IR](简介.md#常用概念) 表示FX Graph或GraphModule，通过对图中的ATen IR变换和分析，可以在不修改原始模型代码的情况下对模型进行灵活修改，例如应用各种图优化Pass操作（算子融合、精度转换、量化等）。

TorchAir提供了一些[FX Pass](简介.md#常用概念)配置项，可以将变换后的ATen IR下沉到aclgraph执行器上，提升算子的执行效率。

## 使用约束

-   本功能支持如下产品：
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
-   本功能仅支持reduce-overhead模式。
-   如果图内存在多流并行计算（即配置[图内多流表达功能（aclgraph）](图内多流表达功能（aclgraph）.md)），本章中所有**reinplace**相关Pass优化将被跳过，换言之开启此类Pass也不会生效。

## 使用方法

该功能通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数介绍参见下表。

```python
import torch_npu, torchair
config = torchair.CompilerConfig()
# 配置图执行模式
config.mode = "reduce-overhead"
# FX Pass配置开关
config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = False
config.debug.aclgraph.disable_reinplace_input_mutated_ops_pass = False
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

**表 1**  参数说明


| 参数名 | 说明 |
| --- | --- |
| aclgraph.disable_reinplace_inplaceable_ops_pass | 布尔类型，是否关闭Pass。该Pass针对模型的中间节点，将节点中包含的[Out-of-place算子](简介.md#section643113494714)（非原地算子）替换为[In-place算子](简介.md#section643113494714)（原地算子），以减少计算过程中的内存搬运，从而提升性能。<br> - False（默认值）：默认打开此Pass。<br> - True：关闭此Pass。<br> **说明**： 多原地算子（修改了多个输入的算子）中，当前仅支持torch_npu.npu_kv_rmsnorm_rope_cache_v2（试验接口，后续版本可能存在变更，暂不支持应用于商用产品中），torch_npu.npu_mla_prolog_v3，torch_npu.npu_add_rms_norm_v2三个算子开启本pass。 |
| aclgraph.disable_reinplace_input_mutated_ops_pass | 布尔类型，是否关闭Pass。该Pass针对模型的原始输入参数，若包含原地操作算子（如KV Cache），Dynamo的Functionalize流程会将原地算子替换为“Out-of-place算子+copy\_算子”，本Pass是该操作的逆向过程，将“Out-of-place算子+copy\_算子”替换为In-place算子，以减少计算过程中的内存搬运，从而提升性能。<br>- False（默认值）：默认打开此Pass。<br>- True：关闭此Pass。<br> **说明**： 多原地算子中，当前仅支持torch_npu.npu_kv_rmsnorm_rope_cache_v2（试验接口，后续版本可能存在变更，暂不支持应用于商用产品中），torch_npu.npu_mla_prolog_v3，torch_npu.npu_add_rms_norm_v2三个算子开启本pass。 |
