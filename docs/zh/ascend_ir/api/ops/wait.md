# wait

## 功能说明

用于在多流间控制时序关系，调用本接口表示当前流需要在传入的tensor对应节点执行结束后再执行。

## 函数原型

```python
wait(tensors: List[torch.Tensor])
```

## 参数说明

|参数|输入/输出|说明|
|--|--|--|
|tensors|输入|List[torch.Tensor]类型，表当前流需要等待的tensor，可以传入多个tensor。|

## 返回值说明

无

## 约束说明

无

## 调用示例

调用示例与[record](record.md)一样。
