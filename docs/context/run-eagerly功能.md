# run-eagerly功能

## 功能简介

TorchAir提供了昇腾NPU图编译后端，对接到不同的图模式执行器（即aclgraph和Ascend IR）。当模型执行出现问题时，无法确定是源于TorchAir本身图变换操作（IR converter、Cache compile等操作）还是图执行器导致的，此时可以**开启run-eagerly**。其可以在图模式执行之前提供Eager模式执行FX graph的能力，通过对比模型前后执行效果，辅助问题定界。

此外，以Eager模式执行FX graph时，还可以**开启静态Kernel编译**，通过算子预先静态编译达到提升网络执行性能的目的。

## 使用约束

-   max-autotune模式下所有功能均不支持与本功能同时开启，即使开启也不生效。
-   reduce-overhead模式下仅[FX Pass配置功能](FX-Pass配置功能.md)、[静态Kernel编译配置](静态Kernel编译配置.md)支持与本功能同时开启。

## 开启run-eagerly

该功能通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数介绍参见下表，完整的示例参考[性能分析案例](性能分析案例.md)。

-   对接到**max-autotune**（Ascend IR）模式的示例如下：

    ```python
    import torch_npu, torchair
    config = torchair.CompilerConfig()
    config.debug.run_eagerly = True
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    opt_model = torch.compile(model, backend=npu_backend)
    ```

-   对接到**reduce-overhead**（aclgraph）模式的示例如下：

    ```python
    import torch_npu, torchair
    config = torchair.CompilerConfig()
    config.mode = "reduce-overhead"
    config.debug.run_eagerly = True
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    opt_model = torch.compile(model, backend=npu_backend)
    ```

**表 1**  参数说明


| 参数名 | 参数说明 |
| --- | --- |
| run_eagerly | 图执行前是否使用Eager模式运行，布尔类型。<br>- False（默认值）：不启动Eager模式，以图模式运行。<br>- True：启动Eager模式运行。 |

## 静态Kernel编译

当采用**reduce-overhead**（aclgraph）模式并开启run-eagerly后，系统将以Eager模式执行FX graph。过程中可通过算子预先静态编译来提升网络执行性能，这种方式称为静态Kernel编译。

开启静态Kernel编译的方法和约束与“[静态Kernel编译配置](静态Kernel编译配置.md)”基本一致，通过在模型编译时指定shape大小，运行时不指定shape大小，以减少运行时开销。

```python
import torch_npu, torchair
config = torchair.CompilerConfig()
# 开启run-eagerly
config.debug.run_eagerly = True
# 采用aclgraph模式
config.mode = "reduce-overhead"
# 开启静态Kernel编译
config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "/path/test"
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```
