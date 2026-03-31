# npugraph\_ex快速上手

npugraph\_ex后端可以提供一个基于torch.compile的简单整图aclgraph加速方案，aclgraph又称为捕获模式，其实现原理请参考《CANN 应用开发指南 \(C&C++\)》中“运行时资源管理\>基于捕获方式构建模型运行实例”章节。

本章将提供aclgraph图模式功能配置的快速上手示例，仅供参考。请根据实际情况自行修改脚本，支持配置的功能参见[功能列表](#功能列表)。

## 使用约束

- 当前npugraph\_ex后端提供的功能均为**试验特性**，后续版本可能存在变更，**暂不支持应用于商用产品中。**
- 主要面向在线推理场景，暂不支持反向流程capture成图、随机数算子capture。
- npugraph\_ex与torch.cuda.CUDAGraph原生接口（参见《PyTorch 原生API支持度》中的“torch.cuda”）功能类似，约束与其保持一致（如不支持stream sync、动态控制流等），此处不再赘述。

## 使用方法

通过配置torch.compile的backend="npugraph\_ex"进行编译。

```python
import torch
import torch_npu

# 自定义Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return torch.add(x, y)

model = Model().npu()
# 基于npugraph_ex backend进行compile
opt_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=False)

# 执行编译后的Model
x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
opt_model(x, y)
```

## 功能列表

npugraph\_ex提供多项功能，包括： [基础功能](#fig2)、[进阶功能](#fig3)和[DFX功能](#fig4)。

其中基础功能通过torch.compile的options参数进行配置，torch.compile为PyTorch原生接口，接口详细介绍请参见[官网](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile)，接口原型如下：

```python
torch.compile(model=None, *, fullgraph=False, dynamic=None, backend='inductor', mode=None, options=None, disable=False)
```

torch.compile参数配置说明参见[表1](#fig1)。

**表 1**  torch.compile参数说明（aclgraph模式）<a id="fig1"></a>

|参数名|PyTorch原生参数说明|aclgraph模式下参数说明|
|--|--|--|
|model|**必选参数**。入图部分的模型或者函数。| 与原生含义一致。|
|fullgraph|可选参数，bool类型。是否捕获整图进行优化。<br>False（缺省值）：非整图优化。<br>True：捕获整图优化。| 建议设置为True，要求将整个函数或模型捕获到一个单一的计算图中。如果编译器遇到无法追踪到该单一图中的代码时（即“图中断”），则会引发错误。|
|dynamic|可选参数，bool类型或None。是否启用动态Shape追踪。<br>None（缺省值）：自动检测是否启用动态Shape追踪。<br>False：不启用动态Shape追踪。<br>True：启用动态Shape追踪。| 与原生含义一致。|
|backend|**必选参数**，后端选择，缺省值为"inductor"。| 需显式传入backend="npugraph_ex"。|
|mode|开销模式，内存开销模式选择，缺省值为None。| 昇腾NPU**暂不支持**。|
|options|优化选项，缺省值为None。| 提供多种基础功能配置，具体参见[基础功能](#fig2)。|
|disable|可选参数，bool类型。是否关闭torch.compile能力。<br>False（缺省值）：开启torch.compile能力。<br>True：关闭torch.compile能力，采用单算子模式。| 与原生含义一致。|

**表 2**  npugraph\_ex基础功能 <a id="fig2"></a>

|功能|功能说明|
|--|--|
|[force_eager功能](./basic/force_eager.md)|图执行前是否使用Eager模式运行。|
|[FX图优化Pass配置功能](./basic/inplace_pass.md)|是否开启FX图优化优化能力。以减少计算过程中的内存搬运，从而提升性能。|
|[FX图算子融合Pass配置功能](./basic/pattern_fusion_pass.md)|是否开启FX图算子融合Pass。该Pass基于已有Aten IR进行融合，从而提升性能。|
|[aclgraph间内存复用功能](./basic/memory_reuse.md)|aclgraph间内存复用功能，支持多种模式。|
|[静态Kernel编译功能](./basic/static_kernel_compile.md)|是否开启静态Kernel编译|
|[冗余算子消除功能](./basic/remove_noop_ops.md)|是否对冗余Kernel进行优化处理|
|[固定权重类输入地址功能](./basic/frozen_parameter.md)|图执行时是否固定权重类输入地址。|
|[重捕获次数限制功能](./basic/capture_limit.md)|设置重捕获次数。|
|[集合通信入图](./basic/communication_graph.md)|实现集合通信算子Ascend Converter，调用torch.compile时默认已支持集合通信算子入图。|
|[Cat算子消除功能](./basic/remove_cat_ops.md)|是否开启Cat算子消除优化以减少内存拷贝和临时张量分配，提升执行性能。|

**表 3**  npugraph\_ex进阶功能 <a id="fig3"></a>

|功能|功能说明|
|--|--|
|[模型编译缓存功能](./advanced/compile_cache.md)|在推理服务和弹性扩容等业务场景中，使用编译缓存可有效缩短服务启动后的首次推理时延。|
|[多流表达功能](./advanced/multi_stream.md)|大模型推理场景下，对于一些可并行的场景，可划分多个stream提升执行效率。|
|[AI-Core和Vector-Core限核功能](./advanced/limit_cores.md)|提供Stream级核数配置，可调整最大AI Core数和Vector Core数，避免算子执行并行度降低。|
|[自定义FX图优化Pass功能](./advanced/post_grad_custom_pass.md)|传入自定义FX Pass函数，该配置可控制自定义Pass在框架内置Pass执行前/后生效。|

**表 4**  npugraph\_ex DFX功能 <a id="fig4"></a>

|功能|功能说明|
|--|--|
[图编译Debug信息保存功能](./dfx/debug_save.md)|通过复用原生DEBUG环境变量TORCH_COMPILE_DEBUG开启日志打印和文件Dump。|
|[算子Data-Dump功能](./dfx/data_dump.md)|是否开启数据dump功能。|

## 问题定界

使用npugraph\_ex后端提供的功能时，如果遇到异常场景，可参考如下问题定界方法，如仍无法解决，请[单击](https://www.hiascend.com/support)联系技术支持。

1. 定界是否是用户脚本问题。

    npugraph\_ex是基于torch.compile扩展aclgraph的功能，因此用户脚本必须先经过PyTorch社区的compile验证才能进行NPU编译，即用户脚本必须先通过下面的尝试后才能继续使用npugraph\_ex。

    ```python
    torch.compile(backend="aot_eager", fullgraph=True)
    ```

    backend="aot\_eager"是PyTorch社区为用户自定义后端做一个简单使用Eager模式直接运行fx.graph的方式。如果这一步出错，那么算子或用户脚本本身就不满足backend="npugraph\_ex"的使用条件。一般来说，此时脚本直接使用torch.npu.graph会报错。

2. 定界是否是aclgraph问题。

    由于npugraph\_ex提供的所有功能均是基于fx.graph做的aclgraph增强体验，因此如果npugraph\_ex出现问题的话，第一步需要定界是aclgraph的问题还是npugraph\_ex增强的功能存在问题，可使用[force\_eager功能](./basic/force_eager.md)辅助定界。

    ```python
    torch.compile(backend="npugraph_ex", fullgraph=True, options={"force_eager": True})
    ```

    如果用户脚本使用force\_eager运行正常，则可能是npugraph\_ex存在问题，否则可能是aclgraph的Runtime底层问题。

## 功能拓展

若您希望额外增加一些自定义的FX图优化功能，可通过[torch.npu.npugraph\_ex.compile\_fx](./api/npugraph_ex/compile_fx.md)接口自定义compiler和backend，然后传入torch.compile进行编译，示例如下：

```python
import torch
from torch._functorch.aot_autograd import aot_module_simplified
import torch_npu

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x = x + y
        return x

# 构建自定义的compiler
def custom_compiler(gm: torch.fx.GraphModule, example_inputs):
    test_options = {
        "clone_input": False
    }
    compiled_graph = torch.npu.npugraph_ex.compile_fx(gm, example_inputs, test_options)
    return compiled_graph

# 构建自定义的backend
def custom_backend(gm: torch.fx.GraphModule, example_inputs):
    return aot_module_simplified(gm, example_inputs, fw_compiler=custom_compiler)
                        
x = torch.ones([2, 2], dtype=torch.int32).npu()
y = torch.ones([2, 2], dtype=torch.int32).npu()
model = torch.compile(Model().npu(), backend=custom_backend, fullgraph=True, dynamic=False)
ret = model(x, y)
```
