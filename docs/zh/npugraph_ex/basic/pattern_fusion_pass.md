# FX图算子融合Pass配置功能

## 功能简介

npugraph\_ex集成了PyTorch原生Pattern能力的算子融合功能，能够通过特定的算子替换规则，使用融合算子替换FX图中多个算子。这种优化可以有效减少部分场景下不必要的下发开销，提高模型执行效率。当与其它图优化策略结合使用时，可通过优化对比来选择最佳方案。

目前框架已提供多种**默认的算子融合Pass**（适用于Deepseek、Long-Cat等网络），参见下表，符合替换规则的算子组合可被替换成对应的融合算子。

**表 1**  已支持的算子融合Pass

|替换规则|对应的融合算子|
|--|--|
|npu_add_rms_norm输出直接作为npu_dynamic_quant（含smooth_scales参数）输入|npu_add_rms_norm_dynamic_quant|
|npu_add_rms_norm输出经flatten(0,1) 后作为npu_dynamic_quant（不含smooth_scales参数）输入，且npu_dynamic_quant输出的scaleOut执行view(-1,1)|npu_add_rms_norm_dynamic_quant（自动处理flatten与view操作）|
|npu_add_rms_norm输出先获取最后一维尺寸h，再经view(-1, h)变形及to(torch.float32)类型转换|npu_add_rms_norm_cast（输出经过view转换）|
|npu_add_rms_norm输出进行to(torch.float32)类型转换|npu_add_rms_norm_cast|
|matmul输出作为transpose输入，transpose参数仅支持(0,1)或者(1,0)，matmul的输入必须是三维|npu_transpose_batchmatmul|
|transpose输出作为matmul输入，matmul输出作为transpose输入，transpose参数仅支持(0,1)或者(1,0)，matmul的输入必须是三维|npu_transpose_batchmatmul（前置transpose）|
|npu_add_rms_norm输出作为npu_quantize输入，npu_add_rms_norm输入尾轴需32B对齐，并满足融合算子npu_add_rms_norm_quant约束条件|npu_add_rms_norm_quant|

另外，用户可通过[register\_replacement](../api/npugraph_ex/register_replacement.md)接口实现自定义算子融合Pass注册（参见接口调用示例），同时需实现[自定义算子入图](../../custom_op_graph/custom_op_graph.md)并自行保证融合规则的正确性。

## 使用约束

- 无论是默认支持的算子融合Pass还是自定义的算子融合Pass，均可由pattern\_fusion\_pass配置。
- 进行npu_transpose_batchmatmul融合时，若matmul的输入shape变化，可能导致融合条件不满足，此时将触发FX图的重新编译生成。
- 融合算子的输出必须被正常使用，同时融合后不再存在的中间结果不能被其他位置引用，否则将无法完成融合。
- 本功能支持的产品型号参见[使用说明](../../overview.md#使用说明)。

## 使用方法

该功能通过npugraph\_ex的options配置，示例如下，仅供参考不支持直接拷贝运行，参数说明参见下表。

```python
import torch
import torch_npu
opt_model = torch.compile(model, backend="npugraph_ex", options={"pattern_fusion_pass": True})
```

**表 2**  参数说明

|**参数名**|**参数说明**|
|--|--|
|pattern_fusion_pass|布尔类型，是否开启算子融合Pass。该Pass基于已有Aten IR进行融合，可提升执行性能。True（默认值）：默认开启此Pass。False：关闭此Pass。|

设置成功后，参考[图编译Debug信息保存功能](../dfx/debug_save.md)开启图编译Debug信息保存，假设原始FX图满足npu\_add\_rms\_norm\_dynamic\_quant的替换规则，可以从Debug信息npugraph\_ex目录中模型前向推理forward子目录的FX图优化输出txt文件看到如下类似的信息，打印信息表明已经存在对应融合算子。

```txt
# No stacktrace found for following nodes
npu_add_rms_norm_dynamic_quant_default = torch.ops.npu.npu_add_rms_norm_dynamic_quant.default(arg2_1, arg1_1, arg0_1, output_mask = [True, True]);  arg2_1 = arg1_1 = arg0_1 = None
getitem_5: "i8[2, 3, 4]" = npu_add_rms_norm_dynamic_quant_default[0]
getitem_6: "f16[2, 3, 4]" = npu_add_rms_norm_dynamic_quant_default[2]
getitem_7: "f32[2, 3]" = npu_add_rms_norm_dynamic_quant_default[3];  npu_add_rms_norm_dynamic_quant_default = None
view_default: "i8[6, 4]" = torch.ops.aten.reshape.default(getitem_5, [6, 4]);  getitem_5 = None
view_default_1: "f32[6, 1]" = torch.ops.aten.reshape.default(getitem_7, [-1, 1]);  getitem_7 = None
return (view_default, view_default_1, getitem_6)
```

## 融合规则

- npu\_add\_rms\_norm\_dynamic\_quant

    ![](../../figures/241127100846395.png)

- npu\_add\_rms\_norm\_dynamic\_quant（自动处理flatten与view操作）

    ![](../../figures/241127100846395-0.png)

- npu\_add\_rms\_norm\_cast（输出经过view转换）

    ![](../../figures/241127100846395-1.png)

- npu\_add\_rms\_norm\_cast

    ![](../../figures/241127100846395-2.png)

- npu\_transpose\_batchmatmul

    ![](../../figures/241127100846395-3.png)

- npu\_transpose\_batchmatmul（前置transpose）

    ![](../../figures/241127100846395-4.png)

- npu\_add\_rms\_norm\_quant

    ![](../../figures/241127100846395-5.png)
