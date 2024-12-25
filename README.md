# 编译

./build.sh -c

> 生成的whl包位于dis目录

# 环境变量

### NPU_INDUCTOR_DEBUG_SINGLE_KERNEL

默认0，设置为1，会在最后生成kernel调用代码时，在前面加入输入tensor信息的打印，并在调用后添加npu同步操作。

### NPU_INDUCTOR_FALLBACK_INT64

默认为1，表示对所有输入类型为int64的ir，进行fallback操作。设置为0则不进行fallback。

### NPU_INDUCTOR_ALWAYS_FALLBACK

默认0，设置为1，会将所有算子的lowering行为变为fallback。

### NPU_INDUCTOR_UNSAFE_CACHE

默认为0，设置为1，会在编译缓存匹配时，只校验对应的so是否存在，而不去检查源码是否一致，用于手动修改并编译kernel替换到整网的需求。

### ASCIR_FORCE_CONTIGUOUS

默认为0，设置为1，则强制在所有的非连续load前，额外插入转连续的逻辑。用于规避不支持非连续输入的问题。

### ASCIR_NOT_READY

默认为0，设置为1，则对ascir、codegen、compiler打桩，不调用任何除了patten匹配和asc构图代码生成外的真实逻辑，用于分析网络patten。

### ASCIR_SUPPORT_CONCAT

默认为0，设置为1，则开启对concat的npu lowering逻辑，因为当前concat限制非常多，无法默认开启。
