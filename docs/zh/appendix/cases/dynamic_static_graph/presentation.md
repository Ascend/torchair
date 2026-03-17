# 动/静态图展示

## Dynamo动/静态图展示

本节提供如下样例test\_compile.py，样例包含parameter、buffer、user\_input输入、scalar输入， 假设dynamic参数采用默认值。

```python
import os
os.environ['TNG_LOG_LEVEL'] = '0'                               # 打开TorchAir的C++层日志
import torch
import torch.nn as nn
import logging
torch._logging.set_logs(dynamo=logging.DEBUG,aot=logging.DEBUG,output_code=True,graph_code=True,recompiles=True)
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger, EVENT_LEVEL
logger.setLevel(logging.DEBUG)                                  # 打开TorchAir python日志

class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(64))    # buffer (全局缓冲区)
        self.linear1 = nn.Linear(128, 64)                        # Linear层 - 其weight和bias自动成为Parameter

    def forward(self, x, scale_factor):
        """
        Args:
            x: user_input tensor (用户输入的tensor)
            scale_factor: scalar输入 (标量输入)
        """
        x = self.linear1(x)
        x = x - self.running_mean
        x = x * scale_factor
        return x

demo_model = DemoModel().npu()
config = CompilerConfig()
# Dump TorchAir优化后的图(py)
config.debug.graph_dump.type = "py"
npu_backend = torchair.get_npu_backend(compiler_config=config)
# 假设dynamic参数采用默认
demo_model = torch.compile(demo_model, dynamic=False, fullgraph=True, backend=npu_backend)

inp = torch.randn(100, 128).npu()
# 输入tensor标记mark_static示例
# torch._dynamo.mark_static(inp)
scale_factor = 100

for i in range(4):
   demo_model(inp, scale_factor)
```

### dynamic=False，Dynamo编译结果为静态图

运行上述脚本test\_compile.py，得到Dynamo生成的原生FX图和TorchAir优化后的图。

-   **原生FX图结构**

    ```txt
    [INFO] TORCHAIR(1630010,python):2025-11-07 15:35:39.502.029 [npu_fx_compiler.py:451]1630010 compiler inputs
    [INFO] TORCHAIR(1630010,python):2025-11-07 15:35:39.502.244 [npu_fx_compiler.py:453]1630010   input 0: FakeTensor(..., device='npu:0', size=(64, 128))
    [INFO] TORCHAIR(1630010,python):2025-11-07 15:35:39.502.597 [npu_fx_compiler.py:453]1630010   input 1: FakeTensor(..., device='npu:0', size=(64,))
    [INFO] TORCHAIR(1630010,python):2025-11-07 15:35:39.502.937 [npu_fx_compiler.py:453]1630010   input 2: FakeTensor(..., device='npu:0', size=(100, 128))
    [INFO] TORCHAIR(1630010,python):2025-11-07 15:35:39.503.269 [npu_fx_compiler.py:453]1630010   input 3: FakeTensor(..., device='npu:0', size=(64,))
    [INFO] TORCHAIR(1630010,python):2025-11-07 15:35:39.503.589 [npu_fx_compiler.py:454]1630010   graph: graph():
        %primals_1 : [num_users=1] = placeholder[target=primals_1]
        %primals_2 : [num_users=1] = placeholder[target=primals_2]
        %primals_3 : [num_users=2] = placeholder[target=primals_3]
        %primals_4 : [num_users=1] = placeholder[target=primals_4]
        %t : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%primals_1,), kwargs = {})
        %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%primals_2, %primals_3, %t), kwargs = {})
        %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%addmm, %primals_4), kwargs = {})
        %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, 100), kwargs = {})
        return (mul, primals_3)
    ```

    FX图中所有输入shape都是固定常量，是一个FX静态图。

-   **TorchAir优化后的图**

    ```python
    # -*- coding: utf-8 -*-
    from torch import tensor
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import get_default_ge_graph
    
    primals_1_0 = ge.Data(index=0, dtype=0, shape=[64, 128], placement="NPU", node_name="primals_1")
    primals_2_0 = ge.Data(index=1, dtype=0, shape=[64], placement="NPU", node_name="primals_2")
    primals_3_0 = ge.Data(index=2, dtype=0, shape=[100, 128], placement="NPU", node_name="primals_3")
    primals_4_0 = ge.Data(index=3, dtype=0, shape=[64], placement="NPU", node_name="primals_4")
    
    # File "/npu/pytorch/test/test.py", line 24, in forward    x = self.linear1(x)
    ## FX Code: t: torch.float32[128, 64]npu:0 = torch.ops.aten.t.default(primals_1: torch.float32[64, 128]npu:0)
    ## FX Code: addmm: torch.float32[100, 64]npu:0 = torch.ops.aten.addmm.default(primals_2: torch.float32[64]npu:0, primals_3: torch.float32[100, 128]npu:0, t: torch.float32[128, 64]npu:0)
    Transpose_0 = ge.Transpose(primals_1_0, [1, 0], node_name="Transpose")
    MatMulV2_0 = ge.MatMulV2(primals_3_0, Transpose_0, None, None, node_name="MatMulV2")
    Mul_0 = ge.Mul(MatMulV2_0, ge.Const(1, dtype=0), node_name="Mul")
    Add_0 = ge.Add(Mul_0, primals_2_0, node_name="Add")
    
    # File "/npu/pytorch/test/test.py", line 25, in forward    x = x - self.running_mean
    ## FX Code: sub: torch.float32[100, 64]npu:0 = torch.ops.aten.sub.Tensor(addmm: torch.float32[100, 64]npu:0, primals_4: torch.float32[64]npu:0)
    Sub_0 = ge.Sub(Add_0, primals_4_0, node_name="Sub")
    
    # File "/npu/pytorch/test/test.py", line 26, in forward    x = x * scale_factor
    ## FX Code: mul: torch.float32[100, 64]npu:0 = torch.ops.aten.mul.Tensor(sub: torch.float32[100, 64]npu:0, 100)
    Mul_1_0 = ge.Mul(Sub_0, ge.Const(100, dtype=0), node_name="Mul_1")
    Cast_0 = ge.Cast(Mul_1_0, dst_type=0, node_name="Cast")
    
    _ = ge.NetOutput([Cast_0], dependencies=[])
    ```

    TorchAir优化后的图Aten IR已经转换为GE IR，输入shape都是固定形状，输入scalar常量值转换成**ge.Const**节点，这是一个静态GE原图。

### dynamic=True，Dynamo编译结果为动态图

上述脚本test\_compile.py改为dynamic=True，得到Dynamo生成的原生FX图和TorchAir优化后的图。

-   **原生FX图结构**

    ```txt
    [INFO] TORCHAIR(1632429,python):2025-11-07 15:37:54.655.141 [npu_fx_compiler.py:451]1632429 compiler inputs
    [INFO] TORCHAIR(1632429,python):2025-11-07 15:37:54.655.275 [npu_fx_compiler.py:453]1632429   input 0: s2
    [INFO] TORCHAIR(1632429,python):2025-11-07 15:37:54.655.426 [npu_fx_compiler.py:453]1632429   input 1: s0
    [INFO] TORCHAIR(1632429,python):2025-11-07 15:37:54.655.558 [npu_fx_compiler.py:453]1632429   input 2: FakeTensor(..., device='npu:0', size=(s0, 128))
    [INFO] TORCHAIR(1632429,python):2025-11-07 15:37:54.655.982 [npu_fx_compiler.py:453]1632429   input 3: FakeTensor(..., device='npu:0', size=(s0, 64))
    [INFO] TORCHAIR(1632429,python):2025-11-07 15:37:54.656.374 [npu_fx_compiler.py:454]1632429   graph: graph():
        %primals_6 : [num_users=1] = placeholder[target=primals_6]
        %primals_3 : [num_users=0] = placeholder[target=primals_3]
        %primals_4 : [num_users=1] = placeholder[target=primals_4]
        %tangents_1 : [num_users=1] = placeholder[target=tangents_1]
        %mul_7 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %primals_6), kwargs = {})
        %t_1 : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%mul_7,), kwargs = {})
        %mm : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%t_1, %primals_4), kwargs = {})
        %t_2 : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%mm,), kwargs = {})
        %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_7, [0], True), kwargs = {})
        %view : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%sum_1, [64]), kwargs = {})
        %t_3 : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%t_2,), kwargs = {})
        return (t_3, view, None, None, None, None)
    ```

    FX图中输入shape的具体维度存在符号，是一个FX动态图。

-   **TorchAir优化后的图**

    ```python
    # -*- coding: utf-8 -*-
    from torch import tensor
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import get_default_ge_graph
    
    primals_1_0 = ge.Data(index=0, dtype=0, shape=[64, 128], placement="NPU", node_name="primals_1")
    primals_2_0 = ge.Data(index=1, dtype=0, shape=[64], placement="NPU", node_name="primals_2")
    primals_3_0 = ge.Data(index=2, dtype=9, shape=[], placement="CPU", node_name="primals_3")
    primals_4_0 = ge.Data(index=3, dtype=0, shape=[-1, 128], placement="NPU", node_name="primals_4")
    Shape_0 = ge.Shape(primals_4_0, dtype=9, node_name="Shape")
    Gather_0 = ge.Gather(Shape_0, 0, node_name="Gather")
    primals_5_0 = ge.Data(index=4, dtype=0, shape=[64], placement="NPU", node_name="primals_5")
    primals_6_0 = ge.Data(index=5, dtype=9, shape=[], placement="CPU", node_name="primals_6")
    
    # File "/npu/pytorch/test/test.py", line 24, in forward    x = self.linear1(x)
    ## FX Code: t: torch.float32[128, 64]npu:0 = torch.ops.aten.t.default(primals_1: torch.float32[64, 128]npu:0)
    ## FX Code: addmm: torch.float32[s0, 64]npu:0 = torch.ops.aten.addmm.default(primals_2: torch.float32[64]npu:0, primals_4: torch.float32[s0, 128]npu:0, t: torch.float32[128, 64]npu:0)
    Transpose_0 = ge.Transpose(primals_1_0, [1, 0], node_name="Transpose")
    MatMulV2_0 = ge.MatMulV2(primals_4_0, Transpose_0, None, None, node_name="MatMulV2")
    Mul_0 = ge.Mul(MatMulV2_0, ge.Const(1, dtype=0), node_name="Mul")
    Add_0 = ge.Add(Mul_0, primals_2_0, node_name="Add")
    
    # File "/npu/pytorch/test/test.py", line 25, in forward    x = x - self.running_mean
    ## FX Code: sub_1: torch.float32[s0, 64]npu:0 = torch.ops.aten.sub.Tensor(addmm: torch.float32[s0, 64]npu:0, primals_5: torch.float32[64]npu:0)
    Sub_0 = ge.Sub(Add_0, primals_5_0, node_name="Sub")
    
    # File "/npu/pytorch/test/test.py", line 26, in forward    x = x * scale_factor
    ## FX Code: mul_4: torch.float32[s0, 64]npu:0 = torch.ops.aten.mul.Tensor(sub_1: torch.float32[s0, 64]npu:0, primals_6: "Sym(s2)")
    Cast_0 = ge.Cast(primals_6_0, dst_type=0, node_name="Cast")
    Mul_1_0 = ge.Mul(Sub_0, Cast_0, node_name="Mul_1")
    Cast_1_0 = ge.Cast(Mul_1_0, dst_type=0, node_name="Cast_1")
    
    _ = ge.NetOutput([Cast_1_0], dependencies=[])
    ```

    TorchAir优化后的图Aten IR已经转换为GE IR，FX图中输入tensor对应的符号位维度转换成了-1，输入scalar符号转换成了ge.Data节点，ge.Data的shape是固定常量，这是一个动态GE原图。

### dynamic=True且对输入mark\_static，Dynamo编译结果为静态图

上述脚本test\_compile.py改为dynamic=True且对输入tensor标记mark\_static，得到Dynamo生成的原生FX图和TorchAir优化后的图。

```python
# 输入tensor标记mark_static示例
inp = torch.randn(100, 128).npu()
torch._dynamo.mark_static(inp)
```

-   **原生FX图结构**

    ```txt
    [INFO] TORCHAIR(1639859,python):2025-11-07 15:45:40.989.451 [npu_fx_compiler.py:451]1639859 compiler inputs
    [INFO] TORCHAIR(1639859,python):2025-11-07 15:45:40.989.644 [npu_fx_compiler.py:453]1639859   input 0: FakeTensor(..., device='npu:0', size=(64, 128))
    [INFO] TORCHAIR(1639859,python):2025-11-07 15:45:40.989.992 [npu_fx_compiler.py:453]1639859   input 1: FakeTensor(..., device='npu:0', size=(64,))
    [INFO] TORCHAIR(1639859,python):2025-11-07 15:45:40.990.312 [npu_fx_compiler.py:453]1639859   input 2: FakeTensor(..., device='npu:0', size=(100, 128))
    [INFO] TORCHAIR(1639859,python):2025-11-07 15:45:40.990.627 [npu_fx_compiler.py:453]1639859   input 3: FakeTensor(..., device='npu:0', size=(64,))
    [INFO] TORCHAIR(1639859,python):2025-11-07 15:45:40.990.955 [npu_fx_compiler.py:453]1639859   input 4: s0
    [INFO] TORCHAIR(1639859,python):2025-11-07 15:45:40.991.110 [npu_fx_compiler.py:454]1639859   graph: graph():
        %primals_1 : [num_users=1] = placeholder[target=primals_1]
        %primals_2 : [num_users=1] = placeholder[target=primals_2]
        %primals_3 : [num_users=2] = placeholder[target=primals_3]
        %primals_4 : [num_users=1] = placeholder[target=primals_4]
        %primals_5 : [num_users=2] = placeholder[target=primals_5]
        %t : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%primals_1,), kwargs = {})
        %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%primals_2, %primals_3, %t), kwargs = {})
        %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%addmm, %primals_4), kwargs = {})
        %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %primals_5), kwargs = {})
        return (mul, primals_3, primals_5)
    ```

    FX图中所有输入shape都是固定常量，但输入scalar值是符号（对应脚本中的scalar输入），是一个FX动态图。

-   **TorchAir优化后的图**

    ```python
    # -*- coding: utf-8 -*-
    from torch import tensor
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import get_default_ge_graph
    
    primals_1_0 = ge.Data(index=0, dtype=0, shape=[64, 128], placement="NPU", node_name="primals_1")
    primals_2_0 = ge.Data(index=1, dtype=0, shape=[64], placement="NPU", node_name="primals_2")
    primals_3_0 = ge.Data(index=2, dtype=0, shape=[100, 128], placement="NPU", node_name="primals_3")
    primals_4_0 = ge.Data(index=3, dtype=0, shape=[64], placement="NPU", node_name="primals_4")
    primals_5_0 = ge.Data(index=4, dtype=9, shape=[], placement="CPU", node_name="primals_5")
    
    # File "/npu/pytorch/test/test.py", line 24, in forward    x = self.linear1(x)
    ## FX Code: t: torch.float32[128, 64]npu:0 = torch.ops.aten.t.default(primals_1: torch.float32[64, 128]npu:0)
    ## FX Code: addmm: torch.float32[100, 64]npu:0 = torch.ops.aten.addmm.default(primals_2: torch.float32[64]npu:0, primals_3: torch.float32[100, 128]npu:0, t: torch.float32[128, 64]npu:0)
    Transpose_0 = ge.Transpose(primals_1_0, [1, 0], node_name="Transpose")
    MatMulV2_0 = ge.MatMulV2(primals_3_0, Transpose_0, None, None, node_name="MatMulV2")
    Mul_0 = ge.Mul(MatMulV2_0, ge.Const(1, dtype=0), node_name="Mul")
    Add_0 = ge.Add(Mul_0, primals_2_0, node_name="Add")
    
    # File "/npu/pytorch/test/test.py", line 25, in forward    x = x - self.running_mean
    ## FX Code: sub: torch.float32[100, 64]npu:0 = torch.ops.aten.sub.Tensor(addmm: torch.float32[100, 64]npu:0, primals_4: torch.float32[64]npu:0)
    Sub_0 = ge.Sub(Add_0, primals_4_0, node_name="Sub")
    
    # File "/npu/d00885544/pytorch/every_test/1107_doc/test.py", line 26, in forward    x = x * scale_factor
    ## FX Code: mul: torch.float32[100, 64]npu:0 = torch.ops.aten.mul.Tensor(sub: torch.float32[100, 64]npu:0, primals_5: "Sym(s0)")
    Cast_0 = ge.Cast(primals_5_0, dst_type=0, node_name="Cast")
    Mul_1_0 = ge.Mul(Sub_0, Cast_0, node_name="Mul_1")
    Cast_1_0 = ge.Cast(Mul_1_0, dst_type=0, node_name="Cast_1")
    
    _ = ge.NetOutput([Cast_1_0], dependencies=[])
    ```

    TorchAir优化后的图Aten IR已经转换为GE IR，FX图中的输入shape都是固定形状，scalar符号输入转换成了ge.Data节点，ge.Data的shape是固定常量，优化后是一个静态GE原图。

## GE动/静态图展示

本章沿用[Dynamo动/静态图展示](#dynamo动静态图展示)提供的样例脚本test\_compile.py。运行该脚本，通过TorchAir的编译日志或Dump的GE build图来判断编译后的图是否完全静态下沉调度。

Dump GE图参见《CANN 环境变量参考》中的“DUMP\_GE\_GRAPH”章节，dump的图文件一般为txt、pbtxt两种格式，txt文件比pbtxt文件多保存了一些属性信息，pbtxt文件可使用Netron等可视化软件打开。

-   txt格式判断：build图中**如果存在graph的\_graph\_unknown\_flag属性值，且取值为true**则为非完全静态下沉调度，否则为完全静态下沉调度。
-   pbtxt：下面分别提供不同场景下的图信息展示，其中GE build图通过可视化的pbtxt格式展示。

### dynamic=False，GE编译结果为静态shape图 <a id="sec1"></a>

-   通过TorchAir编译日志判断编译结果

    ```txt
    [INFO] TORCHAIR(1630010,python):2025-11-07-15:35:41.717.513 [concrete_graph/concrete_graph.cpp:246]1630010 Static FX graph compiled to static Ascend GE graph, graph id: 0
    ```

    “static Ascend GE graph”标识表明这是一个完全静态下沉调度的图。

-   GE build图

    ![](../../../figures/zh-cn_image_0000002481310690.png)

### dynamic=True，GE编译结果为动态shape图 <a id="sec2"></a>

-   通过TorchAir编译日志判断编译结果

    ```txt
    [INFO] TORCHAIR(1632429,python):2025-11-07-15:38:03.235.847 [concrete_graph/concrete_graph.cpp:246]1632429 Dynamic FX graph compiled to dynamic Ascend GE graph, graph id: 0
    ```

-   GE build图

    ![](../../../figures/zh-cn_image_0000002513270567.png)

### dynamic=True且对输入mark\_static，GE编译结果为静态shape图 <a id="sec3"></a>

-   通过TorchAir编译日志判断编译结果

    ```txt
    [INFO] TORCHAIR(1639859,python):2025-11-07-15:45:43.284.466 [concrete_graph/concrete_graph.cpp:246]1639859 Static FX graph compiled to static Ascend GE graph, graph id: 0
    ```

-   GE build图

    ![](../../../figures/zh-cn_image_0000002513390561.png)

本场景与[dynamic=False，GE编译结果为静态shape图](#sec1)均为GE静态shape图，但图结构不同，本图多了一个**Cast节点**。原因是TorchAir Converter会将torch.SymInt（即[dynamic=True且对输入mark\_static，Dynamo编译结果为静态图](#sec3)中原生FX图结构里的**input 4: s0**）转换为DT\_INT64类型，而后续计算是DT\_FLOAT格式，因此需通过Cast将数据从DT\_INT64转换为DT\_FLOAT。

