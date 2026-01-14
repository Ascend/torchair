# TorchAir Python层日志

## 功能简介

TorchAir的Python层日志（图编译过程中的日志信息）主要通过logger.setLevel函数设置，实现Python层功能调试和问题定位。

支持的日志级别如下：

-   logging.DEBUG：日志级别DEBUG。
-   logging.INFO：日志级别INFO。
-   logging.WARNING：日志级别WARNING。
-   logging.ERROR：日志级别ERROR。
-   TorchAir的EVENT\_LEVEL：日志级别EVENT，用于统计TorchAir关键流程耗时。开启后输出ERROR、EVENT日志。

除了EVENT级别是TorchAir自定义的，其他日志级别与Python原生使用方法类似，可参考Python官网[logging模块](https://docs.python.org/3.8/library/logging.html)，logger.setLevel的默认值为“logging.ERROR“。

## 使用方法（Ascend IR图编译）

TorchAir中Ascend IR图编译日志的打印方法如下：

```python
import logging
import torch_npu
from torchair import logger

# 设置Debug日志级别
logger.setLevel(logging.DEBUG)
# 设置EVENT日志级别（TorchAir自定义）
# from torchair.core.utils import EVENT_LEVEL
# logger.setLevel(EVENT_LEVEL)
```

Python侧Debug日志样例如下：

```text
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.250.813 [npu_fx_compiler.py:242]2250956 before sym input optimization, graph is graph():
%arg0_1 : [num_users=1] = placeholder[target=arg0_1]
%arg1_1 : [num_users=1] = placeholder[target=arg1_1]
%arg2_1 : [num_users=1] = placeholder[target=arg2_1]
%scatter_update : [num_users=1] = call_function[target=torch.ops.npu.scatter_update.default](args = (%arg0_1, %arg1_1, %arg2_1, -2), kwargs = {})
return (scatter_update,)
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.251.624 [npu_fx_compiler.py:238]2250956 after sym input optimization, graph is graph():
%arg0_1 : [num_users=1] = placeholder[target=arg0_1]
%arg1_1 : [num_users=1] = placeholder[target=arg1_1]
%arg2_1 : [num_users=1] = placeholder[target=arg2_1]
%scatter_update : [num_users=1] = call_function[target=torch.ops.npu.scatter_update.default](args = (%arg0_1, %arg1_1, %arg2_1, -2), kwargs = {})
return (scatter_update,)
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.252.076 [npu_fx_compiler.py:112]2250956 -------------------
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.252.187 [npu_fx_compiler.py:113]2250956 target: arg0_1
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.253.160 [npu_fx_compiler.py:119]2250956 output Pack(meta:FakeTensor(dtype=torch.float32, size=[1, 1, 2, 8] npu:Tensor(arg0_1:0, dtype=DT_FLOAT, size=[1, 1, 2, 8])))
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.253.478 [npu_fx_compiler.py:112]2250956 -------------------
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.253.601 [npu_fx_compiler.py:113]2250956 target: arg1_1
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.254.196 [npu_fx_compiler.py:119]2250956 output Pack(meta:FakeTensor(dtype=torch.int64, size=[1] npu:Tensor(arg1_1:0, dtype=DT_INT64, size=[1])))
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.254.503 [npu_fx_compiler.py:112]2250956 -------------------
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.254.609 [npu_fx_compiler.py:113]2250956 target: arg2_1
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.255.063 [npu_fx_compiler.py:119]2250956 output Pack(meta:FakeTensor(dtype=torch.float32, size=[1, 1, 1, 8] npu:Tensor(arg2_1:0, dtype=DT_FLOAT, size=[1, 1, 1, 8])))
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.255.330 [npu_fx_compiler.py:112]2250956 -------------------
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.255.433 [npu_fx_compiler.py:113]2250956 target: npu.scatter_update.default
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.255.676 [npu_fx_compiler.py:115]2250956 input 0: Pack(meta:FakeTensor(dtype=torch.float32, size=[1, 1, 2, 8] npu:Tensor(arg0_1:0, dtype=DT_FLOAT, size=[1, 1, 2, 8])))
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.255.830 [npu_fx_compiler.py:115]2250956 input 1: Pack(meta:FakeTensor(dtype=torch.int64, size=[1] npu:Tensor(arg1_1:0, dtype=DT_INT64, size=[1])))
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.255.955 [npu_fx_compiler.py:115]2250956 input 2: Pack(meta:FakeTensor(dtype=torch.float32, size=[1, 1, 1, 8] npu:Tensor(arg2_1:0, dtype=DT_FLOAT, size=[1, 1, 1, 8])))
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.256.042 [npu_fx_compiler.py:115]2250956 input 3: -2
[DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44.297.158 [npu_fx_compiler.py:119]2250956 output Pack(meta:FakeTensor(dtype=torch.float32, size=[1, 1, 2, 8] npu:Tensor(Scatter:0, dtype=DT_FLOAT, size=[1, 1, 2, 8])))
```

## 使用方法（Dynamo编译）

原生Dynamo图编译日志的打印方法如下：

```python
import logging
torch._logging.set_logs(dynamo=logging.DEBUG,aot=logging.DEBUG,output_code=True,graph_code=True)
```

Dynamo侧原生日志：

```text
[2025-02-06 16:46:56,297] [0/0] torch._dynamo.output_graph: [DEBUG] COMPILING GRAPH due to GraphCompileReason(reason='return_value', user_stack=[<FrameSummary file /home/torchair_example/tests/examples/test_scatter_update.py, line 17 in forward>], graph_break=False)
[2025-02-06 16:46:56,300] [0/0] torch._dynamo.output_graph.__graph_code: [DEBUG] TRACED GRAPH
[2025-02-06 16:46:56,300] [0/0] torch._dynamo.output_graph.__graph_code: [DEBUG]  ===== __compiled_fn_0 =====
[2025-02-06 16:46:56,300] [0/0] torch._dynamo.output_graph.__graph_code: [DEBUG]  <eval_with_key>.0 class GraphModule(torch.nn.Module):
[2025-02-06 16:46:56,300] [0/0] torch._dynamo.output_graph.__graph_code: [DEBUG]     def forward(self, L_var_ : torch.Tensor, L_indices_ : torch.Tensor, L_updates_ : torch.Tensor):
[2025-02-06 16:46:56,300] [0/0] torch._dynamo.output_graph.__graph_code: [DEBUG]         l_var_ = L_var_
[2025-02-06 16:46:56,300] [0/0] torch._dynamo.output_graph.__graph_code: [DEBUG]         l_indices_ = L_indices_
[2025-02-06 16:46:56,300] [0/0] torch._dynamo.output_graph.__graph_code: [DEBUG]         l_updates_ = L_updates_
[2025-02-06 16:46:56,300] [0/0] torch._dynamo.output_graph.__graph_code: [DEBUG]
[2025-02-06 16:46:56,300] [0/0] torch._dynamo.output_graph.__graph_code: [DEBUG]         # File: /home/torchair_example/tests/examples/test_scatter_update.py:16, code: output = torch_npu.scatter_update(var, indices, updates, -2)
[2025-02-06 16:46:56,300] [0/0] torch._dynamo.output_graph.__graph_code: [DEBUG]         scatter_update = torch.ops.npu.scatter_update(l_var_, l_indices_, l_updates_, -2);  l_var_ = l_indices_ = l_updates_ = None
[2025-02-06 16:46:56,300] [0/0] torch._dynamo.output_graph.__graph_code: [DEBUG]         return (scatter_update,)
[2025-02-06 16:46:56,300] [0/0] torch._dynamo.output_graph.__graph_code: [DEBUG]
[2025-02-06 16:46:56,300] [0/0] torch._dynamo.output_graph.__graph_code: [DEBUG]
[2025-02-06 16:46:56,301] [0/0] torch._dynamo.output_graph.__graph: [DEBUG] Tabulate module missing, please install tabulate to log the graph in tabular format, logging code instead:
[2025-02-06 16:46:56,301] [0/0] torch._dynamo.output_graph.__graph: [DEBUG] TRACED GRAPH
[2025-02-06 16:46:56,301] [0/0] torch._dynamo.output_graph.__graph: [DEBUG]  ===== __compiled_fn_0 =====
[2025-02-06 16:46:56,301] [0/0] torch._dynamo.output_graph.__graph: [DEBUG]  <eval_with_key>.0 class GraphModule(torch.nn.Module):
[2025-02-06 16:46:56,301] [0/0] torch._dynamo.output_graph.__graph: [DEBUG]     def forward(self, L_var_ : torch.Tensor, L_indices_ : torch.Tensor, L_updates_ : torch.Tensor):
[2025-02-06 16:46:56,301] [0/0] torch._dynamo.output_graph.__graph: [DEBUG]         l_var_ = L_var_
[2025-02-06 16:46:56,301] [0/0] torch._dynamo.output_graph.__graph: [DEBUG]         l_indices_ = L_indices_
[2025-02-06 16:46:56,301] [0/0] torch._dynamo.output_graph.__graph: [DEBUG]         l_updates_ = L_updates_
[2025-02-06 16:46:56,301] [0/0] torch._dynamo.output_graph.__graph: [DEBUG]
[2025-02-06 16:46:56,301] [0/0] torch._dynamo.output_graph.__graph: [DEBUG]         # File: /home/torchair_example/tests/examples/test_scatter_update.py:16, code: output = torch_npu.scatter_update(var, indices, updates, -2)
[2025-02-06 16:46:56,301] [0/0] torch._dynamo.output_graph.__graph: [DEBUG]         scatter_update = torch.ops.npu.scatter_update(l_var_, l_indices_, l_updates_, -2);  l_var_ = l_indices_ = l_updates_ = None
[2025-02-06 16:46:56,301] [0/0] torch._dynamo.output_graph.__graph: [DEBUG]         return (scatter_update,)
[2025-02-06 16:46:56,301] [0/0] torch._dynamo.output_graph.__graph: [DEBUG]
[2025-02-06 16:46:56,301] [0/0] torch._dynamo.output_graph.__graph: [DEBUG]
[2025-02-06 16:46:56,302] [0/0] torch._dynamo.output_graph.__graph_sizes: [DEBUG] TRACED GRAPH TENSOR SIZES
[2025-02-06 16:46:56,302] [0/0] torch._dynamo.output_graph.__graph_sizes: [DEBUG] ===== __compiled_fn_0 =====
[2025-02-06 16:46:56,302] [0/0] torch._dynamo.output_graph.__graph_sizes: [DEBUG] l_var_: (1, 1, 2, 8)
[2025-02-06 16:46:56,302] [0/0] torch._dynamo.output_graph.__graph_sizes: [DEBUG] l_indices_: (1,)
[2025-02-06 16:46:56,302] [0/0] torch._dynamo.output_graph.__graph_sizes: [DEBUG] l_updates_: (1, 1, 1, 8)
[2025-02-06 16:46:56,302] [0/0] torch._dynamo.output_graph.__graph_sizes: [DEBUG] scatter_update: (1, 1, 2, 8)
[2025-02-06 16:46:56,302] [0/0] torch._dynamo.output_graph.__graph_sizes: [DEBUG]
```

