# FAQ

## 本地编译失败，报错libboundscheck等库下载失败

### 问题描述

本地编译失败，报错信息提示libboundscheck等库下载失败。

### 解决方案

1. 优先考虑，配置好本地git环境的网络权限，保证编译时依赖库可以被成功下载。
2. 如果本地环境没有网络权限，可以考虑其他环境下载好指定版本的libboundscheck库的whl包，将本地whl包软链接到cmake指定的路径，即可完成编译。

## 运行时遇到报错“CHECK failed: GeneratedDatabase()->Add(encoded_file_descriptor, size)”

### 问题描述

报错日志信息如下：

```txt
[libprotobuf ERROR descriptor_database.cc:644] File already exists in database: dvpp_tensor_shape.proto
[libprotobuf FATAL descriptor.cc:1371] CHECK failed: GeneratedDatabase()->Add(encoded_file_descriptor, size):
......
torch._dynamo.exc.BackendCompilerFailed: backend='functools.partial(<function _npu_backend at 0xfffdb1f92820>, compiler_config=<torchair.configs.compiler_config.CompilerConfig object at 0xfffa5d17a160>, aot_config=None, custom_decompositions={})' raised:
RuntimeError: CHECK failed: GeneratedDatabase()->Add(encoded_file_descriptor, size):
```

### 原因分析

当前TE版本与系统CANN版本不适配，需要安装对应的CANN版本的TE包。

### 解决方案

找到当前使用的CANN位置，重新安装CANN中的TE包，例如当前CANN安装在/usr/local/Ascend里，那么找到/usr/local/Ascend/cann/aarch64-linux/lib64/te-\*\*\*.whl安装即可。
