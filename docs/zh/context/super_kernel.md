# super\_kernel

## 功能说明

图执行过程中，标记图内能融合为SuperKernel的上下文算子范围，详细功能介绍参见[图内标定SuperKernel范围](图内标定SuperKernel范围.md)。

## 函数原型

```python
super_kernel(scope: str, options: str = '')
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| scope | 输入 | 字符串类型，表示上下文算子被融合的SuperKernel名，相同的scope代表相同的范围，由用户控制。<br>若传入None，表示该范围内的算子不进行SuperKernel融合。 | 必选 |
| options | 输入 | 字符串类型，表示融合的SuperKernel编译选项。默认情况下，系统编译模式采用所有编译选项默认值（参见[表1](图内标定SuperKernel范围.md#table1)）。<br>同时支持用户自定义组合编译选项，配置格式形如"<option1>=<value1>:<option2>=<value2>:<option3>=......"，多个选项时用英文冒号分割。 | 可选 |

## 返回值说明

无

## 约束说明

-   本功能仅支持max-autotune模式，适用于静态图场景。
-   本功能支持如下产品：
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

-   需要注意的是，SuperKernel融合会按网络中算子顺序依次识别能否被融合，**若识别到不可融合的算子**，生成第一段SuperKernel，同时自动跳过该算子进行第二段SuperKernel融合。
-   目前支持SuperKernel融合的通信类算子包括AllReduce、ReduceScatter、AllGather、AlltoAll。

## 调用示例

参考[图内标定SuperKernel范围 > 使用示例](图内标定SuperKernel范围.md#使用示例)。

