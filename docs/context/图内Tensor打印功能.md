# 图内Tensor打印功能

## 功能简介

在图模式下，由于Python原生print函数会触发断图（graph break），导致图模式下无法使用print观察图执行过程中的tensor信息（包含value、shape、dtype）。

TorchAir提供了一个类似原生print特性且又不会断图的打印接口（torchair.ops.npu\_print），方便用户观察图执行过程，以便快速定位问题。

## 使用约束

-   本功能仅支持max-autotune模式。
-   打印为异步打印，打印输出顺序与图中执行顺序一致，与非图内执行的其他输出顺序无关。

    例如图执行后，又在外部调用Python的print打印函数，可能出现图中打印位于Python print之后打印的情况。

-   接口为异步接口，打印数据会占用额外的Device内存同时耗费Device侧执行时间，请合理设置打印的Tensor数据展示量，否则会因内存不足或执行超时导致失败。一般建议打印的数据量在KB级别以下。
-   对于Complex类型的tensor，不支持打印对应value值。
-   对于torch.Tensor类型输入，支持打印的数据类型包括：torch.int8、torch.uint8、torch.int16、torch.int32、torch.int64、torch.uint16、torch.uint32、torch.uint64、torch.float16、torch.float32、torch.float64、torch.bool、torch.bfloat16。其中torch.uint16、torch.uint32、torch.uint64类型的打印需要使用\>=PyTorch 2.3.0的版本，这是PyTorch的原生约束。

## 使用方法

在网络训练/推理脚本中，按需调用torchair.ops.npu\_print接口打印目标Tensor值，接口说明参见[npu\_print](npu_print.md)。

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