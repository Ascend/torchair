# npu\_stream\_switch

## 功能说明

图执行过程中如需开启“**图内多流表达功能**”，可通过本接口指定图内多个算子分发到不同stream做并行计算，提高资源利用率。

## 函数原型

```python
npu_stream_switch(stream_tag: str, stream_priority: int = 0)
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| stream_tag | 输入 | 字符串类型，指定算子执行的目标stream标签。相同的标签代表相同的流，由用户控制。 | 必选 |
| stream_priority | 输入 | int类型，表示切换到stream_tag流的优先级，即Runtime运行时在并发时优先给高优先级的流分配核资源。当前版本为预留参数，建议取默认值0。 | 可选 |

## 返回值说明

无

## 约束说明

无

## 调用示例

-   模式1：参考[图内多流表达功能（aclgraph）> 使用示例](图内多流表达功能（aclgraph）.md#使用示例)。
-   模式2：参考[图内多流表达功能（Ascend-IR）> 使用示例](图内多流表达功能（Ascend-IR）.md#使用示例)。

