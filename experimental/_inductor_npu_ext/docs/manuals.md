# inductor-npu-ext 使用手册

本文档提供 [inductor-npu-ext](../README.md) 的使用指南和常见问题排查方法。

## 目录

- [脚本使能方式](#1-脚本使能方式)
- [调测相关环境变量](#2-调测相关环境变量)
- [调测输出说明](#3-调测输出说明)

## 1. 脚本使能方式

通过在 Python 脚本中导入 inductor-npu-ext 以启用功能。

> 必须在 @torch.compile 装饰的函数或模块被调用之前导入 inductor-npu-ext，否则相关功能将无法启用。

```python
import torch
import torch_npu
import inductor_npu_ext
```

## 2. 调测相关环境变量

### TORCH_COMPILE_DEBUG

**作用：** 启用详细调试日志，以及编译中间产物保存等，参见 [DEBUG 信息说明](#3-DEBUG信息说明)。

**使用方法：**

```bash
export TORCH_COMPILE_DEBUG=1
```

**注意：**
多次执行相同脚本，会因为缓存存在而跳过编译，可以配合 TORCHINDUCTOR_FORCE_DISABLE_CACHES 使用，强制每次执行都重新编译。

---

### TORCHINDUCTOR_FORCE_DISABLE_CACHES

**作用：** 禁用 Inductor 缓存，每次执行都会重新编译。

**使用方法：**

```bash
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
```

**注意：**
会显著增加图启动耗时，实际部署时请勿使用该环境变量。

---

### ASCEND_LAUNCH_BLOCKING

**作用：** 启用 Ascend 内核同步执行，每次kernel下发都会等待完成，便于确定首个报错的 kernel。

**使用方法：**

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

**注意：**
会显著降低下发性能，实际部署时请勿使用该环境变量。

## 3. 调测输出说明

本节结合以具体示例，说明在启用 TORCH_COMPILE_DEBUG=1 后，相关的调试信息和编译产物的输出位置和内容。

**测试脚本：**

> 假定测试脚本为 test.py，内容如下：

```python
import torch
import torch_npu
import inductor_npu_ext

@torch.compile
def test_add_sum(x, y):
    return torch.add(x, y).sum()

x = torch.randn(32, 1024).npu()
y = torch.randn(1, 1024).npu()

out = test_add_sum(x, y)
```

**设置调测环境变量并执行测试脚本：**

```bash
export TORCH_COMPILE_DEBUG=1
python3 test.py
```

### 3.1 关键日志说明

**Generating kernel for ...**
> 该日志表示 inductor-npu-ext 正在为某个融合结构生成 kernel，日志中会包含融合结构的Loop表达。

```
[2026-03-12 19:26:39] [RANK0] [DEBUG] Generating kernel for fused:
# Topologically Sorted Source Nodes: [add, sum_1], Original ATen: [aten.add, aten.sum]
# Source node to ATen node mapping:
#   add => add
#   sum_1 => sum_1
# Graph fragment:
#   %arg0_1 : Tensor "f32[32, 1024][1024, 1]npu" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[1, 1024][1024, 1]npu" = PlaceHolder[target=arg1_1]
#   %add : Tensor "f32[32, 1024][1024, 1]npu"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %sum_1 : Tensor "f32[][]npu"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%add,), kwargs = {})
#   return %sum_1
[2026-03-12 19:26:39] [RANK0] [DEBUG] Codegen [1/1] op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', 0, {})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', c0, {c0: 32768}),
        MemoryDep('arg1_1', c1, {c0: 32, c1: 1024})]
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('npu', torch.float32, size=[], stride=[])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = npu
op0.group.iteration = (1, 32768)
op0.sizes = ([], [32, 1024])
arg0_1_layout = FixedLayout('npu', torch.float32, size=[32, 1024], stride=[1024, 1])
arg1_1_layout = FixedLayout('npu', torch.float32, size=[1, 1024], stride=[1024, 1])
buf0_layout = FixedLayout('npu', torch.float32, size=[], stride=[])
class op0_loop_body:
    var_ranges = {p0: 32, p1: 1024}
    index0 = 1024*p0 + p1
    index1 = p1
    index2 = 0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg1_1', get_index_1)
        add = ops.add(load, load_1)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', add)
        get_index_2 = self.get_index('index2')
        store_reduction = ops.store_reduction('buf0', get_index_2, reduction)
        return None
```

**Road for ...**
> 该日志表示 inductor-npu-ext 正在对某个融合结构进行索引变换优化，尽可能减少循环遍历时的内存连续性。
```
[2026-03-12 19:26:39] [RANK0] [DEBUG] Totally transposed indexings score 0 under (p0, p1)
[2026-03-12 19:26:39] [RANK0] [DEBUG] Finally transposed order is (p0, p1) with score 0
[2026-03-12 19:26:39] [RANK0] [DEBUG] Road for 1024*p0 + p1 from [p0, p1]|[32, 1024]|[1024, 1]|0 to [p0, p1]|[32, 1024]|[1024, 1]|0 is dense
[2026-03-12 19:26:39] [RANK0] [DEBUG] Road for p1 from [p0, p1]|[1, 1024]|[0, 1]|0 to [p0, p1]|[32, 1024]|[1024, 1]|0
[2026-03-12 19:26:39] [RANK0] [DEBUG]   broadcast from [p0, p1]|[1, 1024]|[0, 1]|0 to [p0, p1]|[32, 1024]|[1024, 1]|0
[2026-03-12 19:26:39] [RANK0] [DEBUG] graph0 reads ['arg0_1', 'arg1_1'] and writes ['buf0']
```

**Sync compile ...** / **Async compile ...** / **g++ -shared ...**
> 该日志表示 inductor-npu-ext 正在对某个融合结构生成的 kernel 与 wrapper 代码进行编译。
```
[2026-03-12 19:26:40] [RANK0] [DEBUG] Sync compile for /home/medivh/codes/torchair/experimental/_inductor_npu_ext/tests/.npu_kernels_medivh/autofused_add_sum_31aba041b5d992c026cdfc574b77ec24/0b9cb30a2e017a8ba74972338f6e37fa/wrapper.so
[2026-03-12 19:26:40] [RANK0] [DEBUG] g++ -shared -std=c++17 -fPIC -Wall -O2 -o /home/medivh/codes/torchair/experimental/_inductor_npu_ext/tests/.npu_kernels_medivh/autofused_add_sum_31aba041b5d992c026cdfc574b77ec24/0b9cb30a2e017a8ba74972338f6e37fa/kernel.so /tmp/tmpe2sjcypu.cpp
W0312 19:26:40.470000 797352 site-packages/torch/_inductor/debug.py:507] [0/0] model__0_inference_0 debug trace: /home/medivh/codes/torchair/experimental/_inductor_npu_ext/tests/torch_compile_debug/run_2026_03_12_19_26_39_343839-pid_797352/torchinductor/model__0_inference_0.0
```
**Launch args for  ...**
> 该日志表示 inductor-npu-ext 生成的 kernel 正在执行下发。
```
[WRAPPER] Launch args for autofused_add_sum_31aba041b5d992c026cdfc574b77ec24:
[WRAPPER] block_dim: 24
[WRAPPER] stream: 0x123
[WRAPPER] workspace_size: 1048576
[WRAPPER] workspace: 0x456
[WRAPPER] asc_input0: 0x3fec4140
[WRAPPER] asc_input1: 0x3feb3380
[WRAPPER] asc_output0: 0x3f6efb80
```

**Lowered ...** / **Fallback ... as ...**
> 该日志为进程退出时打印的统计信息，Lowered 表示出现并且 inductor-npu-ext 允许 Lowering 的 aten 算子，Fallback 则表示出现但是不允许被 Lowering 的算子。

> 需要注意的是，Lowered 并不意味着实际参与融合，还需要结合该算子是否实现了 Lowering。而 Fallback 日志则会解释不允许 Lowering 的原因。

```
[2026-03-12 19:26:40] [RANK0] [INFO] Lowered 1x aten.add.Tensor(Tensor(torch.float32, torch.Size([32, 1024]), npu),Tensor(torch.float32, torch.Size([1, 1024]), npu)) -> (Tensor(torch.float32, torch.Size([32, 1024]), npu))
[2026-03-12 19:26:40] [RANK0] [INFO] Lowered 1x aten.sum.default(Tensor(torch.float32, torch.Size([32, 1024]), npu)) -> (Tensor(torch.float32, torch.Size([]), npu))
```


### 3.2 落盘文件说明

调试信息输出位于执行目录下的torch_compile_debug子目录，**带有 `autofused_` 前缀的目录**为 inductor-npu-ext 相关产物，其余均为 inductor 原生产物。

```
├── torch_compile_debug
│   └── run_2026_03_12_19_26_39_343839-pid_797352 # 本地执行的时间戳和进程ID
│       ├── torchdynamo
│       │   └── debug.log
│       └── torchinductor
│           ├── aot_model___0_debug.log
│           └── model__0_inference_0.0 # FX图的编译产物目录，与@torch.compile装饰的函数对应。
│               ├── autofused_add_sum_31aba041b5d992c026cdfc574b77ec24 # inductor-npu-ext 相关产物目录，目录目与融合kernel同名，如果FX图中有多个融合kernel，则会有多个以autofused_开头的目录。
│               │   ├── asc_graph.py # 融合kernel对应的AscIR图结构代码
│               │   ├── asc_kernel.py # 融合kernel对应的Ascendc Kernel实现及Tiling实现
│               │   ├── benchmark.py # 融合kernel的基准测试代码，用于评估融合kernel的性能
│               │   ├── graph.svg # 融合kernel对应的AscIR图的可视化文件，需要安装Graphviz工具与pydot库才能生成
│               │   └── inductor_wrapper.cpp # 融合kernel对应的Inductor Wrapper代码，负责将Inductor传入的参数转换为Ascendc Kernel所需的参数，并调用Ascendc Kernel执行
│               ├── fx_graph_readable.py # 可读的 FX 图
│               ├── fx_graph_runnable.py # 可执行的 FX 图
│               ├── fx_graph_transformed.py # Pass 优化后的 FX 图
│               ├── inductor_provenance_tracking_node_mappings.json # Inductor 节点映射的追踪信息
│               ├── ir_post_fusion.txt # FX 图Lowering后，Schedule融合前的中间表示
│               ├── ir_pre_fusion.txt # FX 图Lowering后的中间表示
│               └── output_code.py # 编译生成的最终可执行代码
```
