# create\_npu\_tensors

## 功能说明

通过一串Device地址创建PyTorch在NPU上的Tensors。

主要使用场景是用于创建大模型中的KV Cache Tensors，所有KV Cache的shape和dtype都一致。

> **说明：** 
>-   LLM-DataDist是大模型分布式集群和数据加速组件，提供了集群KV数据管理能力，以支持全量图和增量图分离部署。
>-   该接口目前适用如下产品：
>    -   <term>Atlas A2 推理系列产品</term>
>    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 函数原型

```python
create_npu_tensors(shape: List[int], dtype: torch.dtype, addresses: List[int]) -> List[torch.Tensor]
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| shape | 输入 | 创建Tensor需要的shape信息，格式是整型的List。 | 是 |
| dtype | 输入 | 创建Tensor需要的dtype信息，格式是torch.dtype。 | 是 |
| addresses | 输入 | 创建Tensor需要的device地址信息，格式是整型的List。 | 是 |

## 返回值说明

正常情况下，返回torch.Tensor的List。异常情况会抛出异常，创建Tensor会失败。

## 约束说明

当前该接口仅限大模型分离部署场景下使用。

## 调用示例

```python
import torch, torch_npu, torchair
kv_tensor_addrs = [1, 2, 3, 4]           # 需要是device返回出来的addr
shape = [4, 1024, 128]
dtype = torch.float16
kv_tensors = torchair.llm_datadist.create_npu_tensors(shape, dtype, kv_tensor_addrs)
```
