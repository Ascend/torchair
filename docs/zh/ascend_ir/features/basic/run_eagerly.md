# run-eagerly功能

## 功能简介

当模型执行出现问题时，无法确定是源于TorchAir本身图变换操作（IR converter、Cache compile等操作）还是图执行器导致的，此时可以**开启run-eagerly**。其可以在GE图模式执行之前提供Eager模式执行FX graph的能力，通过对比模型前后执行效果，辅助问题定界。

## 使用约束

开启该功能后，GE图模式相关的功能配置均不生效。

## 开启run-eagerly

该功能通过[torchair.get\_npu\_backend](../../api/torchair/get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数说明参见下表，完整示例可参考[性能分析案例](../../../appendix/cases/performance_cases.md#性能分析案例)。

```python
import torch_npu, torchair
config = torchair.CompilerConfig()
config.debug.run_eagerly = True
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

**表 1**  参数说明

|参数名|参数说明|
|--|--|
|run_eagerly|图执行前是否使用Eager模式运行，布尔类型。<br>False（默认值）：不启动Eager模式，以图模式运行。<br>True：启动Eager模式运行。|
