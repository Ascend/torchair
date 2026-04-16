# aclgraph间内存复用功能

## 功能简介

本功能主要用于解决如下两种场景问题：

- **场景1**：由于aclgraph本身不支持动态shape，因此执行过程中可能会存在多张aclgraph。

    - 当torch.compile使用**dynamic为False**编译时，每一种shape将对应一张FX graph，同时也对应一张aclgraph。
    - 当torch.compile使用**dynamic为True**编译时，多种shape只要满足规则都只会对应一张FX graph，但不同的shape会对应不同的aclgraph。

    这可能导致多张aclgraph占用多份内存，从而在模型执行时引发 OOM（Out Of Memory）问题。为解决这一问题，**npugraph\_ex提供三种内存复用模式**，可按实际需要选择。

    - **模式一**：用户可在options中显式指定要使用的内存池，后续使用该options的graph均使用该内存池，从而达成多个aclgraph内存复用的效果。
    - **模式二**（默认开启）：torch.compile编译过程中默认将一张FX graph里的多种shape捕获成多张aclgraph，实现同一张FX graph内多张aclgraph间的内存复用。如需关闭，需显式设置。
    - **模式三**（默认开启）：通过对user\_inputs类输入进行内存拷贝实现多张aclgraph间的内存复用。如需关闭，需显式设置。
        - 支持同一张FX graph内多张aclgraph间内存复用。
        - 支持多张FX graph的aclgraph间内存复用，并且必须开启模式一。

            > [!NOTE]说明
            >对于一个大模型其输入一般分为以下三类：
            >- **模型自带的parameter/buffer**： 这部分输入tensor在模型初始化时已生成，基本保持不变。特点是预期输入地址固定，生命周期与模型周期相同。如果shape发生变化或tensor释放，都会被Torch Dynamo Guard识别。
            >- **mutated\_inputs**（如kv-cache输入等原地修改的输入）： 这部分输入tensor通常不与模型本身绑定，大部分是单独申请一组tensor，例如vLLM类推理框架会根据可用内存分配kv-cache。特点是占用内存较高，一般不会在单次推理任务中被释放或发生地址变化。
            >- **user\_inputs**（剩余模型输入）： 这部分输入tensor通常指从模型参数传入的tensor，是FX graph所有输入排除上述两类输入之外的部分。特点是占用内存相对较小，单次推理任务中每次推新的token时都是上一个token推理后的输出tensor，因此这部分tensor地址会持续变化。
            >**模式三针对user\_inputs类输入进行内存拷贝（clone）**，并对clone后的输入进行内存复用。若user\_inputs类输入本身占用内存较大，拷贝后可能导致内存不足。此时需要考虑将clone\_input设置为False；而parameter/buffer类和mutated\_inputs类不进行clone，不做显式持有以延长生命周期，而是直接使用。如果此类输入地址发生变化，mutated\_inputs类将触发Recapture（重新进行aclgraph的Capture，重新捕获/记录输入地址）。

- **场景2**：由于aclgraph是基于固定内存地址执行，因此前次执行的输出Tensor内存会被后续执行覆盖。如需将输出结果长时间保存和使用，可能会引入精度问题。针对该问题，**可配置clone\_output为True对输出结果做克隆**。

## 使用约束

- 开启模式一时，模式二功能会自动关闭，模式三配置后正常生效。模式三可以与模式一、模式二共存。
- 本功能支持的产品型号参见[使用说明](../../overview.md#使用说明)。

## 使用方法

该功能通过npugraph\_ex的options配置，示例如下，仅供参考不支持直接拷贝运行，参数说明参见下表。

```python
import torch
import torch_npu

# dynamic=False时，开启模式一和模式三，模式二自动关闭
graph_pool = torch.npu.graph_pool_handle()
opt_model = torch.compile(model, backend="npugraph_ex", options={"use_graph_pool": graph_pool, "clone_input": True, "clone_output": True}, fullgraph=True, dynamic=False)

# dynamic=True时，开启模式二和模式三
# opt_model = torch.compile(model, backend="npugraph_ex", options={"reuse_graph_pool_in_same_fx": True, "clone_input": True, "clone_output": True}, fullgraph=True, dynamic=True)
```

**表 1**  参数说明

|**参数名**|**参数说明**|
|--|--|
|use_graph_pool（模式一）|tuple类型，用于传入需要使用的内存池。一般通过torch.npu.graph_pool_handle主动创建一个pool。torch.npu.graph_pool_handle是PyTorch原生cuda接口torch.cuda.graph_pool_handle的NPU形式。<br>None（默认值）：默认不传入指定内存池。|
|reuse_graph_pool_in_same_fx（模式二）|布尔类型，是否打开FX graph的内存池复用模式。该模式实现了同一张FX graph捕获出来的不同shape的aclgraph之间的内存复用 。<br>True（默认值）：默认打开模式。<br>False：关闭模式。|
|clone_input（模式三）|布尔类型，是否对aclgraph输入做clone。对mutated_input做基于地址变化的自动Recapture处理，对真正的featuremap_input做clone处理以达成多aclgraph间复用input的效果。当用户需要ref处理省这块内存拷贝时可通过此开关控制。<br>True（默认值）：默认开启对输入的clone。<br>False：不对输入clone。<br>当设置为False时，若运行期间输入Tensor地址发生变化（例如传入新Tensor），框架会使用新数据覆盖到第一次运行时的旧Tensor内存中，以避免图重捕获。如果代码中其他地方引用了该旧Tensor，可能会导致数据不一致。|
|clone_output|布尔类型，是否对aclgraph输出做clone。该功能可以在脚本将输出长时间持有的场景下开启，对aclgraph的输出全部做clone之后再返回用户，可以解决长时间持有的输出被覆写而导致的精度问题。<br>False（默认值）：默认不对输出clone。<br>True：开启对输出的clone。|
