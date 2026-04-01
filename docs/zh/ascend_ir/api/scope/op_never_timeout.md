# op\_never\_timeout

## 功能说明

针对GE图中的算子添加\_op\_exec\_never\_timeout属性，即配置算子不超时，使其不参与超时检测，详细功能介绍参见[图内算子不超时配置功能](../../features/advanced/op_never_timeout.md)。

## 函数原型

```python
op_never_timeout(enable: bool = True)
```

## 参数说明

|参数|输入/输出|说明|
|--|--|--|
|enable|输入|bool类型，表示是否添加_op_exec_never_timeout属性，默认为True，表示添加不超时属性。|

## 返回值说明

无

## 约束说明

- 算子融合场景下，若子算子配置了本功能，其无法继承到新的融合算子节点上。
- with语句块内不支持断图。

## 调用示例

参考[使用示例](../../features/advanced/op_never_timeout.md#使用示例)。
