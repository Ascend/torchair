# 自定义算子入图报错“torch.\_C.\_dispatch\_tls\_local\_exclude\_set”

## 问题现象描述

在自定义算子入图过程中，出现了如下类似的报错：

```bash
assert not torch._C._dispatch_tls_local_exclude_set().has(AssertionError:xx)
```

## 可能原因

该报错为PyTorch原生错误，通常发生在PyTorch算子通过torch.library.Library接口（介绍参考[PyTorch官网](https://docs.pytorch.org/docs/stable/library.html#torch.library.Library)）注册时，同时又没有实现Meta推导函数。

## 处理步骤

参考[自定义算子入图](自定义算子入图.md)章节完成Meta推导函数实现。

