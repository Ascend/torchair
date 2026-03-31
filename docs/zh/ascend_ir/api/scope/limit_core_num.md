# limit\_core\_num

## 功能说明

GE图模式（max-autotune）下可通过本接口配置算子执行时使用的最大AI Core数和Vector Core数。

- 说明1：实际使用的核数可能少于配置的最大核数。
- 说明2：配置的最大核数不能超过AI处理器本身允许的最大AI Core数量与最大Vector Core数量。

本接口实现了**算子级核数配置**，具体功能介绍参见[AI Core和Vector Core限核功能](../../features/advanced/limit_cores.md)。

## 函数原型

```python
limit_core_num(op_aicore_num: int, op_vectorcore_num: int)
```

## 参数说明

|参数|输入/输出|说明|
|--|--|--|
|op_aicore_num|输入|整数类型，表示算子运行时的最大AI Core数，取值范围为[1, max_aicore]|
|op_vectorcore_num|输入|整数类型，表示算子运行时的最大Vector Core数，取值范围为[1, max_vectorcore]。当AI处理器上仅存在AI Core不存在Vector Core时，此时仅支持取值为0。|

## 返回值说明

无

## 约束说明

- with语句块内不支持断图。
- 配置核数不能超过AI处理器本身允许的最大核数，假设最大AI Core数为max\_aicore、最大Vector Core数量为max\_vectorcore，系统默认采用最大核数作为实际运行核数。

    您可通过“CANN软件安装目录/<arch\>-linux/data/platform\_config/<soc\_version\>.ini”文件查看，如下所示，说明AI处理器上存在24个Cube Core，存在48个Vector Core。

    ```txt
    [SoCInfo]
    ai_core_cnt=24
    cube_core_cnt=24
    vector_core_cnt=48
    ```

## 调用示例

参考[使用示例](../../features/advanced/limit_cores.md#使用示例)。
