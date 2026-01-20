# limit\_core\_num

## 功能说明

max-autotune和reduce-overhead模式下，用于配置算子执行时使用的最大AI Core数和Vector Core数置。

- 说明1：实际使用的核数可能少于配置的最大核数。
- 说明2：配置的最大核数不能超过AI处理器本身允许的最大AI Core数量与最大Vector Core数量。

注意，两种模式的底层实现并不一样，max-autotune模式实现了算子级核数配置，reduce-overhead模式实现了Stream级核数配置，详细功能介绍参见[图内设置AI Core和Vector Core核数（Ascend IR）](图内设置AI-Core和Vector-Core核数（Ascend-IR）.md)和[图内设置AI Core和Vector Core核数（aclgraph）](图内设置AI-Core和Vector-Core核数（aclgraph）.md)。

## 函数原型

```python
limit_core_num(op_aicore_num: int, op_vectorcore_num: int)
```

## 参数说明

| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| op_aicore_num | 输入 | 整数类型，表示算子运行时的最大AI Core数，取值范围为[1, max_aicore] | 必选 |
| op_vectorcore_num | 输入 | 整数类型，表示算子运行时的最大Vector Core数，取值范围为[1, max_vectorcore]。<br>当AI处理器上仅存在AI Core不存在Vector Core时，此时仅支持取值为0。 | 必选 |

## 返回值说明

无

## 约束说明

- 本功能支持reduce-overhead模式和max-autotune模式。
- max-autotune模式下，**算子级核数配置优先级高于全局核数配置**，具体参见[图内设置AI Core和Vector Core核数（Ascend IR）](图内设置AI-Core和Vector-Core核数（Ascend-IR）.md)。
- 配置核数不能超过AI处理器本身允许的最大核数，假设最大AI Core数为max\_aicore、最大Vector Core数量为max\_vectorcore，系统默认采用最大核数作为实际运行核数。

    您可通过“CANN软件安装目录/_<arch\>_-linux/data/platform\_config/_<soc\_version\>_.ini”文件查看，如下所示，说明AI处理器上存在24个Cube Core，存在48个Vector Core。

    ```bash
    [SoCInfo]
    ai_core_cnt=24
    cube_core_cnt=24
    vector_core_cnt=48
    ```

## 调用示例

- max-autotune：参考[图内设置AI Core和Vector Core核数（Ascend IR） > 使用示例](图内设置AI-Core和Vector-Core核数（Ascend-IR）.md#使用示例)。
- reduce-overhead：参考[图内设置AI Core和Vector Core核数（aclgraph） > 使用示例](图内设置AI-Core和Vector-Core核数（aclgraph）.md#使用示例)。
