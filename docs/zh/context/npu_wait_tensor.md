# npu\_wait\_tensor

## 功能说明

图执行过程中，控制图内多stream并行计算时序，让算子a等待算子b执行完再执行，详细功能介绍参见[图内多流表达功能（Ascend IR）](图内多流表达功能（Ascend-IR）.md)。

## 函数原型

```python
npu_wait_tensor(self: torch.Tensor, dependency: torch.Tensor) 
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| self | 输入 | Tensor类型，表示等待算子的入参，即算子a（后执行）的任意一个输入Tensor。 | 必选 |
| dependency | 输入 | Tensor类型，表示被等待算子的出参，即算子b（先执行）的任意一个输出Tensor。 | 必选 |

## 返回值说明

返回self本身。

## 约束说明

- 该接口一般与[npu\_stream\_switch](npu_stream_switch.md)配套使用，完成图内多流计算配置。
- 仅支持max-autotune模式。

## 调用示例

参考[图内多流表达功能（Ascend-IR）> 使用示例](图内多流表达功能（Ascend-IR）.md#使用示例)。
