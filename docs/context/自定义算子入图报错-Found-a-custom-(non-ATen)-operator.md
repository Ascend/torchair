# 自定义算子入图报错“Found a custom \(non-ATen\) operator”

## 问题现象描述

使能TorchAir图模式后，出现如下报错：

```bash
RuntimeError: Found a custom (non-ATen) operator that either mutates or its inputs: npu::npu_xp_inplace_custom.. Getting these operators to work with functionalization requires some extra work. For mutable ops you need to register a corresponding out-of-place variant of the op, and you also need to register a Functionalization kernel that performs some boilerplate, telling functionalization to map from the mutable op to the out-of-place op. See a more complete example of how to do this at https://gist.github.com/bdhirsh/7dadbf6296f8f7d1abcf4c482f438aaa. Please file a GitHub issue if you run into any problems.
```

## 可能原因

算子为In-place类算子，但是没有实现函数化转换。

## 处理步骤

参考[自定义算子入图](自定义算子入图.md)章节中In-place算子样例，完成函数化转换（将In-place算子替换为非In-place算子）。

