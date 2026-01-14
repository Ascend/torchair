# 注册并适配Eager模式

完成算子NPU实现后，可对接PyTorch的Eager模式进行适配。Ascend Extension for PyTorch提供了OpPlugin算子插件，用来实现**PyTorch算子注册**和**Eager模式适配。**

本章**仅提供OpPlugin适配的关键步骤说明**，详细的操作请参考《PyTorch 框架特性指南》中的“基于OpPlugin算子适配开发”章节，例如算子yaml配置、算子适配等实现。

1.  [注册PyTorch算子](#注册PyTorch算子)
2.  [基于OpPlugin适配Eager模式](#基于OpPlugin适配Eager模式)

## 注册PyTorch算子

PyTorch官方提供的native\_functions.yaml文件定义了PyTorch Native Functions的具体算子定义和分发细节，定义则通过\*.cpp文件实现。OpPlugin库与原生库类似，也使用yaml文件定义了NPU适配的算子，算子具体适配则存放在\*.cpp文件中。

请确保已按[环境准备](环境准备.md)下载torch\_npu源码，算子的ATen IR定义位于third\_party/op-plugin/op\_plugin/config/op\_plugin\_functions.yaml文件中，在“**custom字段**”下添加[目标PyTorch算子](确定算子原型-0.md)Schema定义：

```python
custom:   
- func: my_inplace(Tensor(a!) x, Tensor y) -> Tensor
  op_api: all_version
```

上述原型定义对应的PyTorch算子为torch.ops.npu.my\_inplace，其中torch.ops是PyTorch算子固定开头，npu为torch\_npu自定义算子库的名称，my\_inplace为自定义算子名。

## 基于OpPlugin适配Eager模式

完成算子NPU实现（Ascend C）和PyTorch算子注册后，需要在PyTorch的Eager模式适配层调用Ascend C算子。

借助OpPlugin插件提供的工程化适配能力，简化Eager模式适配层开发，在third\_party/op-plugin/op\_plugin/config/op\_plugin\_functions.yaml的my\_inplace算子原型注册下，追加“**gen\_opapi**”字段（表示对应可结构化的API）：

```python
- func: my_inplace(Tensor(a!) x, Tensor y) -> Tensor
  op_api: all_version
  gen_opapi:
    out:
      size: y                  # 输出的shape和y相同
      dtype: y                 # 输出的dtype和y相同
    exec: aclnnMyInplace       # 等价于调用aclnnMyInplace接口,入参为x、y
```

- out：表示函数的输出，此字段下面包含size和dtype字段，如果输出的shape和dtype某个入参相同，可以这两个字段配置成该入参的名字。

- exec：配置对应的EXEC\_NPU\_CMD接口，一般指aclnnXxx前缀接口。本样例配置为aclnnMyInplace，aclnn为固定前缀，MyInplace为Ascend C算子名，表示调用Ascend C算子MyInplace实现PyTorch算子my\_inplace。
