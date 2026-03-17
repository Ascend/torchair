# npu\_wait\_tensor

## 功能说明

图执行过程中，控制图内多stream并行计算的时序，使得算子a等待算子b执行完再执行，功能说明参见[多流表达功能](../../features/advanced/multi_stream.md)。

## 函数原型

```python
npu_wait_tensor(self: torch.Tensor, dependency: torch.Tensor) 
```

## 参数说明

|参数|输入/输出|说明|
|--|--|--|
|self|输入|Tensor类型，表示等待算子的入参，即算子a（后执行）的任意一个输入Tensor。|
|dependency|输入|Tensor类型，表示被等待算子的出参，即算子b（先执行）的任意一个输出Tensor。|


## 返回值说明

返回self本身。

## 约束说明

该接口一般与[npu\_stream\_switch](npu_stream_switch.md)配套使用，完成图内多流计算配置。

## 调用示例

参考[使用示例](../../features/advanced/multi_stream.md#使用示例)。

