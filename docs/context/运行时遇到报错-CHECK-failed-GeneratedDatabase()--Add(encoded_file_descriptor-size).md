# 运行时遇到报错“CHECK failed: GeneratedDatabase\(\)-\>Add\(encoded\_file\_descriptor, size\)”

## 问题描述

报错日志信息如下：

```bash
[libprotobuf ERROR descriptor_database.cc:644] File already exists in database: dvpp_tensor_shape.proto
[libprotobuf FATAL descriptor.cc:1371] CHECK failed: GeneratedDatabase()->Add(encoded_file_descriptor, size):
......
torch._dynamo.exc.BackendCompilerFailed: backend='functools.partial(<function _npu_backend at 0xfffdb1f92820>, compiler_config=<torchair.configs.compiler_config.CompilerConfig object at 0xfffa5d17a160>, aot_config=None, custom_decompositions={})' raised:
RuntimeError: CHECK failed: GeneratedDatabase()->Add(encoded_file_descriptor, size):
```

## 原因分析

当前TE版本与系统CANN版本不适配，需要安装对应的CANN版本的TE包。

## 解决方案

找到当前使用的CANN位置，重新安装CANN中的TE包，例如当前CANN安装在/usr/local/Ascend里，那么找到/usr/local/Ascend/cann/aarch64-linux/lib64/te-\*\*\*.whl安装即可。
