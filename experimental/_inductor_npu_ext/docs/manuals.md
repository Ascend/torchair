# inductor-npu-ext 使用手册

本文档提供 [inductor-npu-ext](../README.md) 的使用指南和常见问题排查方法。

## 目录

- [脚本使能方式](#1-脚本使能方式)
- [调测相关环境变量](#2-调测相关环境变量)
- [调测输出说明](#3-调测输出说明)
- [持久化编译缓存](#4-持久化编译缓存)

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

**作用：** torch原生环境变量，启用详细调试日志，以及编译中间产物保存等，参见 [调测输出说明](#3-调测输出说明)。

**使用方法：**

```bash
export TORCH_COMPILE_DEBUG=1
```

**注意：**
多次执行相同脚本，会因为缓存存在而跳过编译，可以配合 TORCHINDUCTOR_FORCE_DISABLE_CACHES 使用，强制每次执行都重新编译。

---

### TORCHINDUCTOR_FORCE_DISABLE_CACHES

**作用：** torch原生环境变量，禁用 Inductor 缓存，每次执行都会重新编译。

**使用方法：**

```bash
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
```

**注意：**
会显著增加图启动耗时，实际部署时请勿使用该环境变量。

---

### ASCEND_LAUNCH_BLOCKING

**作用：** torch_npu原生环境变量，启用 Ascend 内核同步执行，每次kernel下发都会等待完成，便于确定首个报错的 kernel。

**使用方法：**

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

**注意：**
会显著降低下发性能，实际部署时请勿使用该环境变量。

## 3. 调测输出说明

本节结合具体示例，说明在启用 TORCH_COMPILE_DEBUG=1 后，相关的调试日志和编译产物。

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
[2026-03-12 19:26:40] [RANK0] [DEBUG] Sync compile for torchair/experimental/_inductor_npu_ext/tests/.npu_kernels_medivh/autofused_add_sum_31aba041b5d992c026cdfc574b77ec24/0b9cb30a2e017a8ba74972338f6e37fa/wrapper.so
[2026-03-12 19:26:40] [RANK0] [DEBUG] g++ -shared -std=c++17 -fPIC -Wall -O2 -o torchair/experimental/_inductor_npu_ext/tests/.npu_kernels_medivh/autofused_add_sum_31aba041b5d992c026cdfc574b77ec24/0b9cb30a2e017a8ba74972338f6e37fa/kernel.so /tmp/tmpe2sjcypu.cpp
W0312 19:26:40.470000 797352 site-packages/torch/_inductor/debug.py:507] [0/0] model__0_inference_0 debug trace: torchair/experimental/_inductor_npu_ext/tests/torch_compile_debug/run_2026_03_12_19_26_39_343839-pid_797352/torchinductor/model__0_inference_0.0
```

**Cache hint ...** / **Reuse cached kernel ...**
> 该日志表示 inductor-npu-ext 在编译某个融合 kernel 时命中缓存。

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

> 需要注意的是，Lowered 日志并不代表一定参与融合，需要结合该算子是否实现了 Lowering。Fallback 日志则会解释放弃 Lowering 的原因。

```
[2026-03-12 19:26:40] [RANK0] [INFO] Lowered 1x aten.add.Tensor(Tensor(torch.float32, torch.Size([32, 1024]), npu),Tensor(torch.float32, torch.Size([1, 1024]), npu)) -> (Tensor(torch.float32, torch.Size([32, 1024]), npu))
[2026-03-12 19:26:40] [RANK0] [INFO] Lowered 1x aten.sum.default(Tensor(torch.float32, torch.Size([32, 1024]), npu)) -> (Tensor(torch.float32, torch.Size([]), npu))
```


### 3.2 落盘文件说明

调试信息输出位于执行目录下的torch_compile_debug子目录，**带有 `autofused_` 前缀的目录**为 inductor-npu-ext 相关产物，其余均为 inductor 原生产物。

> 融合Kernel的命名格式为autofused_{原始节点类型}_{拓扑结构hash}

```
├── torch_compile_debug
│   └── run_2026_03_12_19_26_39_343839-pid_797352 # 本地执行的时间戳和进程ID
│       ├── torchdynamo
│       │   └── debug.log
│       └── torchinductor
│           ├── aot_model___0_debug.log
│           └── model__0_inference_0.0 # 与FX图对应，torch原生规则命名。
│               ├── autofused_add_sum_31aba041b5d992c026cdfc574b77ec24 # 每个融合kernel独立子目录
│               │   ├── asc_graph.py # 融合kernel的AscIR表达
│               │   ├── asc_kernel.py # Codegen生成的Ascendc Kernel及Tiling实现
│               │   ├── benchmark.py # 融合kernel性能测试文件，用于性能评估
│               │   ├── graph.svg # 可视化AscIR表达，需要安装Graphviz与pydot库
│               │   └── inductor_wrapper.cpp # Wrapper代码，负责Kernel下发
│               ├── fx_graph_readable.py # 可读的FX图
│               ├── fx_graph_runnable.py # 可执行的FX图
│               ├── fx_graph_transformed.py # Pass优化后的FX图
│               ├── inductor_provenance_tracking_node_mappings.json # Inductor 节点映射的追踪信息
│               ├── ir_post_fusion.txt # Schedule融合（水平/垂直/...）后的中间表示
│               ├── ir_pre_fusion.txt # FX图Lowering后的中间表示
│               └── output_code.py # 与原始函数等价的最终可执行代码
```

#### 3.2.1 asc_graph.py
该文件包含融合kernel对应的AscIR的定义与Codegen调用。直接执行该文件，可以触发Ascendc Codegen的完整过程。

**建议的使用方式：** 如果对应的融合Kernel在codegen过程中发生错误，可执行该文件，复现Codegen过程的问题。
```bash
python3 asc_graph.py
```

#### 3.2.2 asc_kernel.py
该文件包含融合kernel对应的Ascendc Kernel实现、Tiling实现，以及Jit编译调用逻辑。直接执行该文件，可以触发包含Tiling代码编译的Ascendc完整编译过程，并将结果保存模型执行时的编译缓存。

> 注意，缓存命中时，该文件不会生成。

**建议的使用方式：** 通常不直接使用该文件，如果Kernel执行出现了错误，优先通过benchmark.py尝试复现问题。

> 如果只在整网执行下才可触发，可以修改该文件中的AscendC源码，添加AscendC print或者dump，观察UB中的数据以定位问题原因。执行该文件**会直接修改本地缓存**，下次模型执行生效。
```bash
python3 asc_kernel.py
```

#### 3.2.3 inductor_wrapper.cpp
该文件包含融合kernel对应的Inductor Wrapper代码，负责将Inductor传入的参数转换为Ascendc Kernel所需的参数，并负责内存申请、Tiling计算及Kernel下发。

> 注意，缓存命中时，该文件不会生成。

**建议的使用方式：** 通常不直接使用该文件，如果怀疑Host下发存在问题，优先通过benchmark.py尝试复现问题并修改其中的`cpp_wrapper`实现进行验证。

> 如果问题只在模型执行时才可触发，则可以修改该文件中的C++代码，文件开头包含完整的gcc编译命令，执行该命令进行编译，**会直接修改本地缓存**，下次模型执行生效。

#### 3.2.4 graph.svg
该文件为融合kernel对应的AscIR表达的可视化文件，便于开发者理解融合kernel的结构。
> 需要在模型执行前安装Graphviz工具与pydot库生成，否则不会生成该文件。

**建议的使用方式：** 使用浏览器打开该文件，便于直观地查看融合kernel的结构。

#### 3.2.5 benchmark.py
该文件包含融合kernel的基准测试代码，用于评估融合kernel性能，包含融合Kernel在模型中的输入Tensor构造、Kernel编译与调用、Profiling采集等。

> 对于动态shape的融合kernel，benchmark.py中的输入Tensor为其中一个具体的shape配置，无法覆盖所有输入配置的性能表现，仅供参考。

> 性能测试时，采用相同输入多次执行的方式，L2缓存命中逐步提高，性能数据会呈现从慢（L2命中率低）到快（L2命中率高）的趋势。

**建议的使用方式：** 通常用于观察融合Kernel的性能数据，指导Codegen逻辑的优化或者验证优化效果。该文件接受一个参数输入，参数可以不传或者传入`e2e`。

```
python3 benchmark.py
```
不传入e2e时，执行该文件会执行AscendC代码编译到Profiling采集的完整流程。如果您了解AscendC，可以尝试手工优化其中的AscendC Kernel实现，并观察性能变化。通常用于指导AscIR的Codegen策略优化或快速验证修改方案。

```
python3 benchmark.py e2e
```

传入e2e参数时，执行该文件会执行AscIR表达到Profiling采集的完整流程。如果您优化了AscIR的Codegen实现或修复了Codegen的bug，可以通过该流程验证效果。

## 4. 持久化编译缓存
inductor-npu-ext 会将融合Kernel编译结果进行缓存落盘。缓存在多进程、多线程、进程内的多个子进程以及同一进程内的相同pattern间共享，通过文件锁保证安全性。

缓存保存在**执行目录下**以`.npu_kernels_{用户名}`开头的目录中，每个融合kernel一个子目录，融合kernel子目录下，为以Codegen结果hash命名的二级子目录，用于处理不同软件版本间的Codegen实现差异。缓存匹配包含两部分：
- 融合结构的明文拓扑（不受节点名、符号名等冗余信息影响）
- Codegen结果的hash值

只有两者都匹配时才会命中缓存。

```
├── .npu_kernels_root
│   └── autofused_add_sum_31aba041b5d992c026cdfc574b77ec24
│       └── 0b9cb30a2e017a8ba74972338f6e37fa # Codegen结果的hash值
│           ├── asc_kernel.py
│           ├── compile.lock
│           ├── inductor_wrapper.cpp
│           ├── kernel.so
│           └── wrapper.so
```

假如您在调试某个编译问题，不希望缓存生效，可以在每次执行前手动删除该目录或其中的某个子目录。