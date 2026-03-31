# limit\_core\_num

## 功能说明

aclgraph模式下，可通过本接口配置算子执行时使用的最大AI Core数和Vector Core数。

- 说明1：实际使用的核数可能少于配置的最大核数。
- 说明2：配置的最大核数不能超过AI处理器本身允许的最大AI Core数量与最大Vector Core数量。

本接口实现了**Stream级核数配置**，具体功能介绍参见[AI Core和Vector Core限核功能](../../advanced/limit_cores.md)。

## 函数原型

```python
limit_core_num(op_aicore_num: int, op_vectorcore_num: int)
```

## 参数说明

|参数|输入/输出|说明|
|--|--|--|
|op_aicore_num|输入|整数类型，表示算子运行时的最大AI Core数，取值范围为[1, max_aicore]。|
|op_vectorcore_num|输入|整数类型，表示算子运行时的最大Vector Core数，取值范围为[1, max_vectorcore]。当AI处理器上仅存在AI Core不存在Vector Core时，此时仅支持取值为0。|

## 返回值说明

无

## 约束说明

with语句块内不支持断图。

## 调用示例

接口调用参见[使用示例](../../advanced/limit_cores.md#使用示例)。
