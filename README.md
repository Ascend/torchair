# 编译

./build.sh -c

> 生成的whl包位于dis目录

# 环境变量

### NPU_INDUCTOR_DEBUG_SINGLE_KERNEL

默认0，设置为1，会在最后生成kernel调用代码时，在前面加入输入tensor信息的打印，并在调用后添加npu同步操作。

### ASCIR_NOT_READY

默认为0，设置为1，则对ascir、codegen、compiler打桩，不调用任何除了patten匹配和asc构图代码生成外的真实逻辑，用于分析网络patten。

### ASCIR_SUPPORT_CONCAT

默认为0，设置为1，则开启对concat的npu lowering逻辑，因为当前concat限制非常多，无法默认开启。
