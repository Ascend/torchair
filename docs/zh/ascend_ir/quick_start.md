# GE图模式快速上手

GE图模式一般通过TorchAir的CompilerConfig属性**mode="max-autotune"**开启（该模式是系统默认模式），其将FX图转换为Ascend IR图，并通过GE图引擎实现图编译和执行。

本章将提供GE图模式功能配置的快速上手示例，仅供参考。请根据实际情况自行修改脚本，支持配置的功能参见后续章节。

## 使用方法

```python
# 导包（必须先导torch_npu再导torchair）
import torch
import torch_npu
import torchair

# Patch方式实现集合通信入图（可选）
from torchair import patch_for_hcom
patch_for_hcom()

# 自定义Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.add(x, y)

model = Model().npu()
# 图执行模式默认为max-autotune
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)

# 基于TorchAir backend进行compile
opt_model = torch.compile(model, backend=npu_backend)

# 执行编译后的Model
x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
opt_model(x, y)
```

## torch.compile

torch.compile为PyTorch原生接口，接口详细介绍请参见官网[Link](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile)，接口原型如下：

```python
torch.compile(model=None, *, fullgraph=False, dynamic=None, backend='inductor', mode=None, options=None, disable=False)
```

GE图模式下的torch.compile参数配置说明参见[表1](#fig1)。

**表 1**  torch.compile参数说明（Ascend IR模式）<a id="fig1"></a>

|参数名|PyTorch原生参数说明|Ascend IR模式下参数说明|
|--|--|--|
|model|**必选参数**。入图部分的模型或者函数。|与原生含义一致。|
|fullgraph|可选参数，bool类型。是否捕获整图进行优化。<br>False（缺省值）：非整图优化。<br>True：捕获整图优化。|与原生含义一致。|
|dynamic|可选参数，bool类型或None。是否启用动态Shape追踪。<br>None（缺省值）：自动检测是否启用动态Shape追踪。<br>False：不启用动态Shape追踪。<br>True：启用动态Shape追踪。|与原生含义一致。|
|backend|**必选参数**，后端选择，缺省值为"inductor"。|如需使用TorchAir提供的后端，需通过torchair.get_npu_backend获取并显式传入。通过**compiler_config参数**配置图模式功能，支持的功能项参见[表2](#fig2)。|
|mode|开销模式，内存开销模式选择，缺省值为None。|昇腾NPU**暂不支持**。|
|options|优化选项，缺省值为None。|昇腾NPU**暂不支持**。|
|disable|可选参数，bool类型。是否关闭torch.compile能力。<br>False（缺省值）：开启torch.compile能力。<br>True：关闭torch.compile能力，采用单算子模式。|与原生含义一致。|


**表 2**  CompilerConfig功能项<a id="fig2"></a>

|分类|说明|功能项|
|--|--|--|
|debug|配置debug调试类功能，配置形式为config.debug.xxx。|[图结构dump功能](./features/basic/graph_dump.md)<br>[算子data-dump功能（Eager模式）](./features/basic/data_dump_eager.md)<br>[run-eagerly功能](./features/basic/run_eagerly.md)<br>[算子Converter支持度导出功能](./features/advanced/converter_export.md)|
|export|配置离线导图相关功能，配置形式为config.export.xxx。|[Dynamo导图功能](./features/advanced/dynamo_export.md)|
|dump_config|配置图模式下dump功能，配置形式为config.dump_config.xxx。|[算子data-dump功能（Ascend-IR）](./features/advanced/data_dump.md)|
|fusion_config|配置图融合相关功能，配置形式为config.fusion_config.xxx。|[算子融合规则配置功能（fusion_switch_file）](./features/advanced/fusion_switch_file.md)|
|experimental_config|配置各种试验功能，配置形式为config.experimental_config.xxx。|[冗余算子消除功能（Ascend-IR）](./features/basic/remove_noop_ops.md)<br>[FX图算子融合Pass配置功能Ascend-IR](./features/basic/pattern_fusion_pass.md)<br>[固定权重类输入地址功能（Ascend-IR）](./features/advanced/frozen_parameter.md)<br>[图模式编译节点遍历选项](./features/advanced/topology_sorting_strategy.md)<br>[计算与通信并行功能](./features/advanced/cc_parallel.md)<br>[算子在线编译选项](./features/advanced/jit_compile.md)<br>[RefData类型转换功能](./features/advanced/ref_data.md)<br>[Tiling调度优化功能](./features/advanced/tiling_schedule_optimize.md)<br>[View类算子优化功能](./features/advanced/view_optimize.md)<br>[动静子图拆分场景性能优化](./features/advanced/static_model_ops_lower_limit.md)|
|inference_config|配置推理相关功能，配置形式为config.inference_config.xxx。|[动态shape图分档执行功能](./features/advanced/dynamic_gears_merge_policy.md)|
|ge_config|配置GE图引擎提供的功能，配置形式为config.ge_config.xxx。|[图编译统计信息导出功能](./features/advanced/export_compile_stat.md)<br>[单流执行功能](./features/advanced/single_stream.md)<br>[图编译多级优化选项](./features/advanced/oo_level.md)<br>[算子融合规则配置功能（optimization_switch）](./features/advanced/optimization_switch.md)<br>[AI-Core和Vector-Core限核功能（Ascend-IR）](./features/advanced/limit_cores.md)|


