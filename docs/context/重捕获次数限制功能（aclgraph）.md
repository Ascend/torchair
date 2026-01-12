# 重捕获次数限制功能（aclgraph）

## 功能简介

> **须知：** reduce-overhead模式下的功能为试验特性，后续版本可能存在变更，暂不支持应用于商用产品中。

reduce-overhead模式下，由于aclgraph本身不支持动态shape，因此执行过程中可能会因为输入shape的变化而多次重新捕获aclgraph。

重新捕获的过程会带来额外的性能开销和Device侧资源消耗，尤其是频繁的重捕获，可能导致资源不足。

本功能提供了一个配置项，用于控制单张FX图中不同shape的重捕获次数阈值。当超过设置的阈值时，后续的所有执行将全部回退到Eager模式。

## 使用约束

-   本功能支持如下产品：
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
-   本功能仅支持reduce-overhead模式。

## 使用方法

该功能配置示例如下，仅供参考不支持直接拷贝运行，参数介绍参见下表。

```python
import torch_npu, torchair
config = torchair.CompilerConfig()
# 配置图执行模式
config.mode = "reduce-overhead"
# 重捕获次数限制设置
config.debug.aclgraph.static_capture_size_limit = 64
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

**表 1**  参数说明


| 参数名 | 说明 |
| --- | --- |
| aclgraph.static_capture_size_limit | int类型，最小值1，最大值为9223372036854775807。<br>64（默认值）：默认允许的重捕获次数为64。<br>注意：当重捕获时，NPU设备上stream、event、memory等资源超限时，会因为资源不足退出流程，可能无法触发到最大值。 |

