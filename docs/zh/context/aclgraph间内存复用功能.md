# aclgraph间内存复用功能

## 功能简介

>**须知：** reduce-overhead模式下的功能为试验特性，后续版本可能存在变更，暂不支持应用于商用产品中。

-   **场景1：**

    reduce-overhead模式下，由于aclgraph本身不支持动态shape，因此执行过程中可能会存在多张aclgraph。

    -   当torch.compile使用dynamic为False编译时，每一种shape将对应一张FX graph，同时也对应一张aclgraph。
    -   当torch.compile使用dynamic为True编译时，多种shape只要满足规则都只会对应一张FX graph，但不同的shape会对应不同的aclgraph。

    这可能导致多张aclgraph占用多份内存，从而在模型执行时引发 OOM（Out Of Memory）问题。为解决这一问题，TorchAir提供**三种内存复用模式**，可按需选择。

    -   **模式一**：用户可在config中显式指定要使用的内存池，后续使用该config的所有graph均可使用该内存池，从而实现多个aclgraph的内存复用。
    -   **模式二**（默认开启）：torch.compile编译过程中默认将一张FX graph里的多种shape捕获成多张aclgraph，实现同一张FX graph内多张aclgraph间的内存复用。如需关闭，需显式设置。
    -   **模式三**（默认开启）：通过对user\_inputs类输入进行内存拷贝实现多张aclgraph间的内存复用。如需关闭，需显式设置。

        -   支持同一张FX graph内多张aclgraph间内存复用。
        -   支持多张FX graph的aclgraph间内存复用，并且必须开启模式一。

        > **说明：** 
        >对于一个大模型其输入一般分为以下三类：
        >-   模型自带的parameter/buffer：
        >    这部分输入tensor在模型初始化时已生成，基本保持不变。特点是预期输入地址固定，生命周期与模型周期相同。如果shape发生变化或tensor释放，都会被Torch Dynamo Guard识别。
        >-   mutated\_inputs（如kv-cache输入等原地修改的输入）：
        >    这部分输入tensor通常不与模型本身绑定，大部分是单独申请一组tensor，例如VLLM类推理框架会根据可用内存分配kv-cache。特点是占用内存较高，一般不会在单次推理任务中被释放或发生地址变化。
        >-   user\_inputs（剩余模型输入）：
        >    这部分输入tensor通常指从模型参数传入的tensor，是FX graph所有输入排除上述两类输入之外的部分。特点是占用内存相对较小，单次推理任务中每次推新的token时都是上一个token推理后的输出tensor，因此这部分tensor地址会持续变化。
        >   **模式三针对user\_inputs类输入进行内存拷贝（clone）**，并对clone后的输入进行内存复用；而parameter/buffer类和mutated\_inputs类不进行clone，不做显式持有以延长生命周期，而是直接使用，如果此类输入地址发生变化，将触发Recapture（重新进行aclgraph的Capture，重新捕获/记录输入地址）。

-   **场景2：**

    由于aclgraph基于固定内存地址执行，因此前次执行的输出Tensor内存可能会被后续执行覆盖。如需将输出结果长时间保存和使用，可能会引入精度问题。针对该问题，可使用[Clone](Clone.md)接口**对指定输出结果做克隆**。

## 使用约束

-   本功能支持如下产品：
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
-   本功能仅支持reduce-overhead模式。
-   需要注意的是，开启模式一时，模式二功能会自动关闭，模式三配置正常生效。模式三可以与模式一、模式二共存。

## 使用方法

该功能通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数说明如下表。

```python
import torch_npu, torchair
config = torchair.CompilerConfig()
# 配置图执行模式
config.mode = "reduce-overhead"
# 开启模式一
config.aclgraph_config.use_custom_pool = torch.npu.graph_pool_handle()
# 关闭模式二
config.debug.aclgraph.disable_mempool_reuse_in_same_fx = True
# 关闭模式三
config.debug.aclgraph.clone_input = False
# 主动开启输出内存拷贝
config.debug.aclgraph.enable_output_clone = True
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

**表 1**  参数说明


| 参数名 | 说明 |
| --- | --- |
| aclgraph_config.use_custom_pool（模式一） | tuple类型，用于传入需要使用的内存池。一般通过torch.npu.graph_pool_handle主动创建一个pool。<br> None（默认值）：默认不传入指定内存池。<br>**说明**： torch.npu.graph_pool_handle是PyTorch原生cuda接口torch.cuda.graph_pool_handle的NPU形式。 |
| debug.aclgraph.disable_mempool_reuse_in_same_fx（模式二） | 布尔类型，是否关闭FX graph的内存池复用模式。该模式实现同一张FX graph捕获出来的不同shape的aclgraph之间的内存复用 。<br>- False（默认值）：默认打开模式。<br>- True：关闭模式。 |
| debug.aclgraph.clone_input（模式三） | 布尔类型，是否对aclgraph的user_inputs类输入做内存拷贝（clone）。该模式实现多aclgraph间（同一张FX graph内多张aclgraph间，或者多张FX graph的aclgraph间）的输入内存复用。<br>- True（默认值）：默认对输入clone。<br>- False：关闭对输入的clone。 |
| debug.aclgraph.enable_output_clone | 布尔类型，是否对aclgraph的输出做内存拷贝（clone）。当开启内存池复用时，若输出长时间持有的情况下可开启本功能，对aclgraph的输出全部做clone再返回，以解决长时间持有的输出被覆写而导致的精度问题。<br>- False（默认值）：默认不对输出clone。<br>- True：开启对输出的clone。 |

