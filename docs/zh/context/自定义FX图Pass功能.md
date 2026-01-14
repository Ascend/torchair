# 自定义FX图Pass功能

## 功能简介

用户可通过自定义图优化Pass，将其注册到TorchAir中，达到修改PyTorch FX图的目的。

## 使用约束

TorchAir约定FX Pass的函数签名如下：

```python
def _(gm, example_inputs, config: torchair.CompilerConfig) -> None
```

-   gm：表示AOT（Ahead-of-Time）编译后的GraphModule类对象，gm.graph为其FX图。
-   example\_inputs：表示AOT（Ahead-of-Time）编译后的GraphModule对象的FakeTensor类型输入，通常不需要使用。
-   config：表示TorchAir编译后端创建的[CompilerConfig类](CompilerConfig类.md)对象，用于Pass感知完整编译选项。

FX Pass原地修改gm对象，任何返回值都会被忽略。对于无法处理的异常情况，应当抛出异常。需要确保不抛出异常时，处理后的FX图是正确的：即其执行结果与修改前的FX图完全一致。

## 使用方法

本章以实现**图内多流并行计算功能**为例，给出自定义Pass编写示例。示例仅供参考，实际业务场景中请按需自行调整代码。

假设有如下网络脚本，目标是指定mm、abs算子在一条新的流上执行，并控制时序让sub算子在abs算子之后执行。

```python
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mm = torch.mm(x, x)
        abs = torch.abs(mm)
        add = torch.add(abs, 1)
        sub = torch.sub(x, mm)
        return add, sub
```

1.  编写自定义Pass，样例如下：

    ```python
    def _custom_pre_pass(gm, example_inputs, config: torchair.CompilerConfig):
        fx_graph = gm.graph
        for node in fx_graph.nodes:        
            if node.op == "call_function" and node.target == torch.ops.aten.mm.default:
                with fx_graph.inserting_before(node):
                    # 在torch.ops.aten.mm.default节点前加入torch.ops.air.scope_enter.default节点
                    # torch.ops.air.scope_enter和torch.ops.air.scope_exit范围内的节点将在新的流stream_1上执行
                    fx_graph.call_function(torch.ops.air.scope_enter.default, args=(
                        ["_user_stream_label"], ["stream_1"]))
    
            if node.op == "call_function" and node.target == torch.ops.aten.abs.default:
                with fx_graph.inserting_after(node):
                    # 在torch.ops.aten.abs.default节点(对应脚本中的torch.abs算子)后插入torch.ops.air.record.default节点
                    record_node = fx_graph.call_function(torch.ops.air.record.default, args=())
                with fx_graph.inserting_after(record_node):
                    fx_graph.call_function(torch.ops.air.scope_exit.default, args=())
    
            # 在torch.ops.aten.sub.Tensor节点(对应脚本中的torch.sub算子)前插入torch.ops.air.wait.default节点
            # 表示需要等待torch.ops.air.record.default前的节点执行完，torch.ops.aten.sub.Tensor才能执行
            if node.op == "call_function" and node.target == torch.ops.aten.sub.Tensor:
                with fx_graph.inserting_before(node):
                    fx_graph.call_function(torch.ops.air.wait.default, args=([record_node],))
    ```

    该Pass的功能是在torch.ops.aten.mm.default和torch.ops.aten.abs.default节点前后分别插入torch.ops.air.scope\_enter.default和torch.ops.air.scope\_exit.default节点，使得指定范围内的节点在流“stream\_1”上执行。并在torch.ops.aten.abs.default节点后插入torch.ops.air.record.default节点，在torch.ops.aten.sub.Tensor节点前插入torch.ops.air.wait.default节点，使得控制时序让sub算子在abs算子之后执行。

2.  将自定义Pass注册到TorchAir使其生效。

    当前TorchAir的编译选项通过torchair.CompilerConfig类传递，需要开启post\_grad\_custom\_pre\_pass和post\_grad\_custom\_post\_pass两个阶段的自定义Pass注册。开关开启方式如下：

    ```python
    import torch
    import torch_npu
    import torchair
    import logging
    from torchair import logger
    # 设置Debug日志级别
    logger.setLevel(logging.DEBUG)
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, x):
            mm = torch.mm(x, x)
            abs = torch.abs(mm)
            add = torch.add(abs, 1)
            sub = torch.sub(x, mm)
            return add, sub
    
    # 自定义Pass修改FX图
    def _custom_pre_pass(gm, example_inputs, config: torchair.CompilerConfig):
        fx_graph = gm.graph
        for node in fx_graph.nodes:        
            if node.op == "call_function" and node.target == torch.ops.aten.mm.default:
                with fx_graph.inserting_before(node):
                    # 在torch.ops.aten.mm.default节点前加入torch.ops.air.scope_enter.default节点
                    # torch.ops.air.scope_enter和torch.ops.air.scope_exit范围内的节点将在新的流stream_1上执行
                    fx_graph.call_function(torch.ops.air.scope_enter.default, args=(
                        ["_user_stream_label"], ["stream_1"]))
    
            if node.op == "call_function" and node.target == torch.ops.aten.abs.default:
                with fx_graph.inserting_after(node):
                    # 在torch.ops.aten.abs.default节点(对应脚本中的torch.abs算子)后插入torch.ops.air.record.default节点
                    record_node = fx_graph.call_function(torch.ops.air.record.default, args=())
                with fx_graph.inserting_after(record_node):
                    fx_graph.call_function(torch.ops.air.scope_exit.default, args=())
    
            # 在torch.ops.aten.sub.Tensor节点(对应脚本中的torch.sub算子)前插入torch.ops.air.wait.default节点
            # 表示需要等待torch.ops.air.record.default前的节点执行完，torch.ops.aten.sub.Tensor才能执行
            if node.op == "call_function" and node.target == torch.ops.aten.sub.Tensor:
                with fx_graph.inserting_before(node):
                    fx_graph.call_function(torch.ops.air.wait.default, args=([record_node],))
    
    config = torchair.CompilerConfig()
    # 将自定义Pass注册到TorchAir中
    config.post_grad_custom_pre_pass = _custom_pre_pass
    # 可选，在TorchAir已有的FX图优化Pass执行后 再执行自定义Pass
    # config.post_grad_custom_post_pass = _custom_pre_pass
    
    # 可选，使用reduce-overhead模式执行
    # config.mode = "reduce-overhead" 
    
    # 可选，max-autotune模式下 dump图结构到./test目录
    config.debug.graph_dump.type = "pbtxt"
    config.debug.graph_dump.path = "./graph_dump"
    # 可选，max-autotune模式下关闭算子融合功能，方便在Profiling文件中观测分流及时序控制结果
    config.ge_config.optimization_switch = "AutomaticUbFusion:off"
    
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = Model().npu()
    opt_model = torch.compile(model, backend=npu_backend)
    x = torch.randn([3, 3]).npu()
    opt_model(x)
    
    # 可选，打印Profiling文件到./prof目录
    experimental_config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level2)
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU,
                        torch_npu.profiler.ProfilerActivity.CPU],
            with_stack=True,
            record_shapes=False,
            profile_memory=False,
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1,
                                                skip_first=0),
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof")) as prof:
    
            opt_model(x)
    
    prof.step()   
    ```

    **表 1**  参数说明

    
    | 参数名 | 说明 |
	| --- | --- |
	| post_grad_custom_pre_pass | TorchAir本身内置了部分FX图优化Pass，该配置控制自定义Pass在内置Pass执行前生效。<br>传入自定义Pass函数。 |
	| post_grad_custom_post_pass | TorchAir本身内置了部分FX图优化Pass，该配置控制自定义Pass在内置Pass执行后生效。<br>传入自定义Pass函数。 |

3.  检查Pass是否生效。
    -   打开Debug日志（参考[TorchAir Python层日志](TorchAir-Python层日志.md)），查看修改后的FX图是否有插入torch.ops.air.scope\_enter.default和torch.ops.air.scope\_exit.default、torch.ops.air.record.default、torch.ops.air.wait.default新节点，以及插入的位置是否正确。

    -   在max-autotune模式下，可以通过dump出的图结构（参考[算子输入输出dump功能（图模式）](算子输入输出dump功能（图模式）.md)）来查看图上对应的算子时序是否符合预期。
    -   可以通过Profiling文件（参考[性能分析案例](性能分析案例.md)）来查看算子的分流以及时序是否符合预期。

