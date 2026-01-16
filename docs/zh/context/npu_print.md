# npu\_print

## 功能说明

在图执行过程中，打印执行脚本中目标Tensor信息。该接口类似原生的print接口，但不会导致断图，详细功能介绍参见[图内Tensor打印功能](图内Tensor打印功能.md)。

## 函数原型

```python
npu_print(*args, summarize_size=3, tensor_detail=False)
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| *args | 输入 | 位置入参，其中每个入参支持的数据类型为torch.Tensor、str、bool、float、int等基本类型，且其中至少包含一个Tensor类型输入。 | 是 |
| summarize_size | 输入 | Tensor每个维度打印的数据元素个数，默认值为3。<br>  - 取值为-1：打印全部数据元素。<br>  - 取值为正整数且大于等于（Tensor维度的最大dim/2）时：打印全部数据元素。<br>  - 取值为正整数且小于（Tensor维度的最大dim/2）时：对于Tensor每个维度，起始位置打印summarize\_size个数据元素，末端位置打印summarize\_size个数据元素，中间元素以“...”表示。<br>打印结果示例参见[约束说明](#约束说明)。 | 否 |
| tensor_detail | 输入 | Tensor是否打印数据shape和dtype信息，默认值为False。<br>  - 取值为False：仅打印Tensor value。<br>  - 取值为True：输出包含shape以及dtype信息。<br>打印结果示例如下：`tensor([0 1 2 ... 7 8 9], shape=[10], dtype=torch.int64)` | 否 |

## 返回值说明

无

## 约束说明

- 本功能仅支持max-autotune模式。

-   打印为异步打印，打印输出顺序与图中执行顺序一致，与非图内执行的其他输出顺序无关。

    例如图执行后，又在外部调用Python的print打印函数，可能出现图中打印位于Python print之后打印的情况。

- 接口为异步接口，打印数据会占用额外的Device内存同时耗费Device侧执行时间，请合理设置打印的Tensor数据展示量，否则会因内存不足或执行超时导致失败。一般建议打印的数据量在KB级别以下。

- 对于Complex类型的tensor，不支持打印对应value值。

- 对于torch.Tensor类型输入，支持打印的数据类型包括：torch.int8、torch.uint8、torch.int16、torch.int32、torch.int64、torch.uint16、torch.uint32、torch.uint64、torch.float16、torch.float32、torch.float64、torch.bool、torch.bfloat16。其中torch.uint16、torch.uint32、torch.uint64类型的打印需要使用\>=PyTorch 2.3.0的版本，这是PyTorch的原生约束。

-   Tensor每个维度打印多个元素的示例：假设shape为[10, 10]的Tensor，summarize\_size=3。
    ```
    [[1 2 3 …… 8 9 10]
    [1 2 3 …… 8 9 10]
    [1 2 3 …… 8 9 10]
     ……
    [1 2 3 …… 8 9 10]
    [1 2 3 …… 8 9 10]
    [1 2 3 …… 8 9 10]]
    ```

## 调用示例

```python
import torch
import torch_npu, torchair

@torch.compile(backend="npu", fullgraph=True)
def hello_tensor(x):
    torchair.ops.npu_print("hello, tensor:", x)

@torch.compile(backend="npu", fullgraph=True)
def hello_tensor_detail(x):
    torchair.ops.npu_print("hello, tensor_detail:", x, tensor_detail=True)

v = torch.arange(10).npu()
hello_tensor(v)         
# 打印结果为"hello, tensor: [0 1 2 ... 7 8 9]"

hello_tensor_detail(v)         
# 打印结果为"hello, tensor_detail: tensor([0 1 2 ... 7 8 9], shape=[10], dtype=torch.int64)"
```

