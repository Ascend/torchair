# npugraph\_ex API列表

本章介绍了npugraph\_ex场景下功能配置可能涉及的Python API，接口列表如下。

## torch.npu.npugraph_ex接口列表

接口名|接口说明|
|--|--|
|[compile_fx](./npugraph_ex/compile_fx.md)|获取编译后的可执行FX图对象，可通过该接口自定义后端，以实现用户自定义的特性。|
|[register_replacement](./npugraph_ex/register_replacement.md)|将自定义算子融合规则注册到TorchAir框架中，在FX图编译后对图进行算子融合优化。|

## torch.npu.npugraph_ex.inference接口列表

接口名|接口说明|
|--|--|
|[cache_compile](./inference/cache_compile.md)|实现aclgraph模式下模型编译缓存，降低成图编译耗时。|
|[readable_cache](./inference/readable_cache.md)|实现aclgraph模式下模型编译缓存时，通过本接口读取封装后的func函数缓存文件compiled_module（格式不限，如py、txt）。|

## torch.npu.npugraph_ex.scope接口列表

接口名|接口说明|
|--|--|
|[limit_core_num](./scope/limit_core_num.md)|图执行过程中，指定图范围内的算子执行时最大的AI Core数和Vector Core数。|


