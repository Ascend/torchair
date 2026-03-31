# npu\_stream\_switch

## 功能说明

通过本接口指定图内多个算子分发到不同stream做并行计算，提高资源利用率，实现[多流表达功能](../../features/advanced/multi_stream.md)。

## 函数原型

```python
npu_stream_switch(stream_tag: str, stream_priority: int = 0)
```

## 参数说明

|参数|输入/输出|说明|
|--|--|--|
|stream_tag|输入|字符串类型，指定算子执行的目标stream标签。相同的标签代表相同的流，由用户控制。|
|stream_priority|输入|int类型，表示切换到stream_tag流的优先级，即Runtime运行时在并发时优先给高优先级的流分配核资源。当前版本为预留参数，建议取默认值0。|

## 返回值说明

无

## 约束说明

with语句块内不支持断图。

## 调用示例

参考[使用示例](../../features/advanced/multi_stream.md#使用示例)。
