# GE图模式API列表

本章介绍了GE图模式场景下功能配置可能涉及的Python API，接口列表如下。

## torchair接口列表

|接口名|接口说明|
|--|--|
|[CompilerConfig类](./torchair/compiler_config.md)|该类用于构造传入torch.compiler backend的配置。|
|[dynamo_export](./torchair/dynamo_export.md)|用于导出由TorchAir生成的离线图（air格式）。|
|[get_compiler](./torchair/get_compiler.md)|获取能够在NPU上运行的图编译器。|
|[get_npu_backend](./torchair/get_npu_backend.md)|获取能够在NPU上运行的图编译后端npu_backend，可作为backend参数传入torch.compile。|
|[use_internal_format_weight](./torchair/use_internal_format_weight.md)|将模型中的权重weight转成TorchAir定义的内部私有格式。|
|[register_fx_node_ge_converter](./torchair/register_fx_node_ge_converter.md)|将自定义算子注册到TorchAir框架中。|
|[patch_for_hcom](./torchair/patch_for_hcom.md)|针对PyTorch 2.1版本中不支持入图的集合通信算子提供的补丁函数，实现部分集合通信算子入图。|
|[register_replacement](./torchair/register_replacement-0.md)|将自定义算子融合规则注册到TorchAir中，在FX图编译后对图进行算子融合优化。|

## torchair.ge接口列表

|接口名|接口说明|
|--|--|
|[DataType类](./ge/datatype.md)|数据类型的枚举类，提供了GE的data type定义，方便实现converter函数时调用。|
|[Format类](./ge/format.md)|数据格式的枚举类，提供了GE的data format定义，方便实现converter函数时调用。|
|[Tensor类](./ge/tensor.md)|提供Tensor定义，用于算子入图的converter函数入参类型声明。|
|[TensorSpec类](./ge/tensorspec.md)|提供TensorSpec定义，表示算子在Meta推导过程中得到的性能，当前用于算子入图的converter函数入参类型声明。|
|[Const](./ge/Const.md)|算子converter中的构图元素，表示一个Const节点，即图中的常量值。|
|[Cast](./ge/Cast.md)|算子converter中的构图元素，表示一个Cast节点，即图中Tensor的类型转换方法。|
|[Clone](./ge/Clone.md)|算子Converter中的构图元素，表示一个Clone节点，该节点可实现图上任意单个Tensor的拷贝。|
|[custom_op](./ge/custom_op.md)|基于算子原型（IR）实现算子converter函数，完成PyTorch IR与GE IR的转换，方便自定义算子入图。|

## torchair.inference接口列表

|接口名|接口说明|
|--|--|
|[cache_compile](./inference/cache_compile.md)|实现GE图模式场景下模型编译缓存，降低成图编译耗时。|
|[readable_cache](./inference/readable_cache.md)|实现GE图模式场景下模型编译缓存时，通过本接口读取封装后的func函数缓存文件compiled_module（格式不限，如py、txt）。|
|[set_dim_gears](./inference/set_dim_gears.md)|将一张图划分为不同档位，使一张图能支持多Batch，提升网络执行性能。|

## torchair.ops接口列表

|接口名|接口说明|
|--|--|
|[npu_print](./ops/npu_print.md)|图执行过程中，打印执行脚本中目标tensor值。|
|[npu_fused_infer_attention_score](./ops/npu_fused_infer_attention_score.md)|图模式场景下的FlashAttention算子，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。|
|[npu_fused_infer_attention_score_v2](./ops/npu_fused_infer_attention_score_v2.md)|图模式场景下的增强版FlashAttention算子，支持全量和增量计算场景。|
|[record](./ops/record.md)|torchair.ops.record用于显式地在当前Stream上下发一个任务，其返回值可以被torchair.ops.wait等待。|
|[wait](./ops/wait.md)|用于在多流间控制时序关系，torchair.ops.wait表示当前流需要在传入的tensor对应的节点执行结束后，再继续执行。|

## torchair.scope接口列表

|接口名|接口说明|
|--|--|
|[npu_stream_switch](./scope/npu_stream_switch.md)|图执行过程中，指定图内多个算子分发到不同stream做并行计算。|
|[npu_wait_tensor](./scope/npu_wait_tensor.md)|图执行过程中，控制图内多stream并行计算时序。|
|[super_kernel](./scope/super_kernel.md)|图执行过程中，标记图内能融合为SuperKernel的上下文算子范围。|
|[limit_core_num](./scope/limit_core_num.md)|图执行过程中，指定图范围内的算子执行时最大的AI Core数和Vector Core数。|
|[op_never_timeout](./scope/op_never_timeout.md)|针对GE图中算子添加_op_exec_never_timeout属性，即配置算子不超时，使其不参与超时检测。|
|[data_dump](./scope/data_dump.md)|图执行过程中，dump指定图内范围内的算子输入输出信息。|

## torchair.llm_datadist接口列表

|接口名|接口说明|
|--|--|
|[create_npu_tensors](./llm_datadist/create_npu_tensors.md)|通过一串Device地址创建PyTorch在NPU上的Tensors。主要用于创建大模型中的KV Cache Tensors，所有KV Cache的shape和dtype都一致。|
