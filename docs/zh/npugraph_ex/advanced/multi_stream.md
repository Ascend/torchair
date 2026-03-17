# 多流表达功能

## 功能简介

大模型推理场景下，对于一些可并行的场景，可以划分多个stream提升执行效率。通过在脚本中指定每个算子的执行stream，将原本需要串行的多个算子分发到不同stream做并行计算，多个stream上的计算形成overlap，从而降低整体计算耗时。

对于并行来说，包含如下两种：

-   计算与计算并行：一般是基于数据依赖关系，分析出可以并行的多条计算分支，指定stream并行。
-   计算与通信并行：一般是针对没有数据依赖的通信操作，提前使用通信资源执行通信任务。

本功能主要处理**aclgraph间资源并发**，尤其针对Cube计算资源未完全使用的场景。若Cube计算资源已完全使用，不建议开启本功能，可能会造成额外的调度，从而导致原计算性能劣化。

## 使用约束

本功能支持如下产品：

-   Atlas A3 训练系列产品/Atlas A3 推理系列产品
-   Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 使用方法

1.  用户自行分析模型脚本中可进行并行计算的算子。
2.  开启多流表达。

    使用如下with语句块，语句块内下发的算子切换至“stream”参数指定的流计算，语句块外的算子使用默认流计算。

    ```python
    with torch.npu.stream(stream: torch.npu.Stream):
    ```

3.  （可选）控制并行计算的时序。

    通过torch.npu.Event\(\)、torch.npu.Event.record\(\)、torch.npu.Event.wait\(\)系列原生接口实现时序控制。

4.  （可选）延长内存释放时机。

    Eager模式场景下，脚本中如果涉及多stream内存复用，一般会调用PyTorch的tensor.record\_stream原生接口延迟内存释放。

5.  （可选）配置限核。

    参考4.3.3章节配置，可以防止出现性能达不到预期，卡死等情况出现。

## 使用示例

```python
import torch
import torch_npu

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        stream = torch.npu.Stream()
        event = torch.npu.Event()
        mul_res = torch.mul(t, 5)
        add_res = torch.add(mul_res, 2)
        event.record()
        with torch.npu.stream(stream):
            event.wait(stream)
            relu_res = torch.relu(add_res)
            add_res.record_stream(stream)
        return relu_res

model = Model().npu()
opt_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=False)
x = torch.randn([3, 3]).npu()
res = opt_model(x)
print(f"res = {res}")
```

