# 算子Converter支持度导出功能

## 功能简介

PyTorch模型使能图模式前，需要先识别[FX](https://pytorch.org/docs/main/fx.html)图中IR是否有对应Converter实现。若有，表示算子支持接入Ascend IR计算图；否则不支持入图。对于不支持入图的算子，请根据实际情况进行Converter补齐，具体操作请参考[Converter补齐](https://gitcode.com/ascend/torchair/blob/master/CONTRIBUTING.md#converter%E8%A1%A5%E9%BD%90)。

本功能可以导出算子详细信息，包括算子名、算子Converter支持度、算子调用次数等。

## 使用约束

本功能仅支持max-autotune模式。

## 使用方法

该功能通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数介绍参见[表1](#table1)。

```python
import torch_npu, torchair
config = torchair.CompilerConfig()
# 导出图中ATen算子信息和相关配置
config.debug.fx_summary.type = "csv"
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

**表 1**  参数说明 <a name="table1"></a>


| 参数名 | 说明 |
| --- | --- |
| fx_summary.type | 指定导出的文件类型，字符串类型。默认为None，不导出图中的ATen算子信息。<br>当前仅支持csv格式。 |
| fx_summary.skip_compile | 是否跳过Ascend IR图编译，以FX图Eager方式执行。bool类型。<br>- True（默认值）：跳过Ascend IR图编译，以FX图Eager方式执行。适用于模型Converter不全，图模式不能正常执行的场景，该场景下能收集完整的FX信息。<br>- False：采用Ascend IR图编译。适用于模型支持以图模式执行且想收集FX信息的场景。 |

> **说明：** 
>在正式运行图模式时，要**删除或注释**config.debug.fx\_summary.type = "csv"这行代码，否则将无法正确使能图模式。

## 产物说明

功能开启后，默认在当前执行路径下生成summary\_$\{timestamp\}.csv文件，文件样例如[表2](#table2)所示。

**表 2**  fx\_summary信息 <a name="table2"></a>


| 目标函数 | 函数类型 | 支持状态 | 调用次数 | 输入统计 | 输出统计 |
| --- | --- | --- | --- | --- | --- |
| aten.as_strided.default | aten | 未实现 | 36 | 24次：(float16(12, 1, 512, 64), [12, 1, 512, 64], [64, 196608, 768, 1])<br>12次：(float16(12, 1024, 64), [12, 2, 768, 64], [65536, 16384, 64, 1]) | 24次：float16(12, 1, 512, 64)<br>12次：float16(12, 2, 768, 64) |
| aten.native_layer_norm.default | aten | 部分支持 | 62 | 62次：(float16(1, s0, 4096), [4096], float16(4096,), float16(4096,), 1e-05) | 62次：(float16(1, s0, 4096), float32(1, s0, 1), float32(1, s0, 1)) |
| aten.add.Tensor | aten | 已支持 | 120 | 60次：(float16(1, s0, 4096), float16(1, s0, 4096))<br>30次：(float16(1, s0, 16384), 1)<br>30次：(float16(1, s0, 16384), 1.0) | 60次：float16(1, s0, 4096)<br>60次：float16(1, s0, 16384) |
| <built-in function getitem> | builtin | 已支持 | 62 | 62次：((float16(1, s0, 4096), float32(1, s0, 1), float32(1, s0, 1)), 0) | 62次：float16(1, s0, 4096) |

-   目标函数+函数类型：表示算子名和归属类型。
-   支持状态：
    -   未实现：该算子Converter未实现，不支持入图。
    -   已支持：该算子实现Converter，支持入图。
    -   部分支持：该算子在部分场景实现Converter，部分场景支持入图。

-   调用次数：表示算子在模型中调用的次数。
-   输入统计：表示算子输入dtype、shape等信息。
-   输出统计：表示算子输出dtype、shape等信息。
