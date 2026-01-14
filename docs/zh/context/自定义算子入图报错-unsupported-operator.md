# 自定义算子入图报错“unsupported operator”

## 问题现象描述

使能TorchAir图模式后，出现如下报错：

```bash
torch._dynamo.exc.Unsupported: unsupported operator: npu.custom.default (see https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.64r4npvq0w0 for how to fix)
```

## 可能原因

算子没有实现Meta推导函数，无法入图。

## 处理步骤

参考[自定义算子入图](自定义算子入图.md)章节完成Meta推导函数实现。
