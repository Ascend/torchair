# inductor-npu-ext 使用手册

本文档提供 [inductor-npu-ext](../README.md) 的使用指南和常见问题排查方法。

## 目录

- [脚本使能方式](#1-脚本使能方式)
- [调测相关环境变量](#2-调测相关环境变量)
- [调测输出说明](#3-调测输出说明)
- [持久化编译缓存](#4-持久化编译缓存)
- [运行时异常说明](#5-运行时异常说明)

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

---

### TORCHINDUCTOR_NPU_EXT_CACHE_DIR

**作用：** 指定 inductor-npu-ext 缓存目录的根路径，缓存目录默认为 `/tmp/.npu_kernels_{用户名}` 。

**注意**，通常你不应该使用该环境变量，实践上，在以下场景可能需要修改默认缓存目录：

- 默认路径被映射在多个OS共享的文件系统上，失去文件锁保护可能造成缓存损坏。通过为每个OS设置不同的缓存目录，避免该问题。
- 调试过程中希望手动管理缓存，例如，保留某些缓存目录以复现问题或降低新环境的首次编译耗时。

**使用方法：**

```bash
export TORCHINDUCTOR_NPU_EXT_CACHE_DIR=/path/to/cache
```

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

```text
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

```text
[2026-03-12 19:26:39] [RANK0] [DEBUG] Totally transposed indexings score 0 under (p0, p1)
[2026-03-12 19:26:39] [RANK0] [DEBUG] Finally transposed order is (p0, p1) with score 0
[2026-03-12 19:26:39] [RANK0] [DEBUG] Road for 1024*p0 + p1 from [p0, p1]|[32, 1024]|[1024, 1]|0 to [p0, p1]|[32, 1024]|[1024, 1]|0 is dense
[2026-03-12 19:26:39] [RANK0] [DEBUG] Road for p1 from [p0, p1]|[1, 1024]|[0, 1]|0 to [p0, p1]|[32, 1024]|[1024, 1]|0
[2026-03-12 19:26:39] [RANK0] [DEBUG]   broadcast from [p0, p1]|[1, 1024]|[0, 1]|0 to [p0, p1]|[32, 1024]|[1024, 1]|0
[2026-03-12 19:26:39] [RANK0] [DEBUG] graph0 reads ['arg0_1', 'arg1_1'] and writes ['buf0']
```

**Sync compile ...** / **Async compile ...** / **g++ -shared ...**
> 该日志表示 inductor-npu-ext 正在对某个融合结构生成的 kernel 与 wrapper 代码进行编译。

```text
[2026-03-12 19:26:40] [RANK0] [DEBUG] Sync compile for torchair/experimental/_inductor_npu_ext/tests/.npu_kernels_medivh/autofused_add_sum_31aba041b5d992c026cdfc574b77ec24/0b9cb30a2e017a8ba74972338f6e37fa/wrapper.so
[2026-03-12 19:26:40] [RANK0] [DEBUG] g++ -shared -std=c++17 -fPIC -Wall -O2 -o torchair/experimental/_inductor_npu_ext/tests/.npu_kernels_medivh/autofused_add_sum_31aba041b5d992c026cdfc574b77ec24/0b9cb30a2e017a8ba74972338f6e37fa/kernel.so /tmp/tmpe2sjcypu.cpp
W0312 19:26:40.470000 797352 site-packages/torch/_inductor/debug.py:507] [0/0] model__0_inference_0 debug trace: torchair/experimental/_inductor_npu_ext/tests/torch_compile_debug/run_2026_03_12_19_26_39_343839-pid_797352/torchinductor/model__0_inference_0.0
```

**Cache hint ...** / **Reuse cached kernel ...**
> 该日志表示 inductor-npu-ext 在编译某个融合 kernel 时命中缓存。

**Launch args for  ...**
> 该日志表示 inductor-npu-ext 生成的 kernel 正在执行下发。

```text
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
> 该日志为进程退出时打印的统计信息，Lowered 表示出现并且 inductor-npu-ext 允许 Lowering 的 aten 算子，Fallback 则表示出现但是不允许被Lowering 的算子。
> 需要注意的是，Lowered 日志并不代表一定参与融合，需要结合该算子是否实现了 Lowering。Fallback 日志则会解释放弃 Lowering 的原因。

```text
[2026-03-12 19:26:40] [RANK0] [INFO] Lowered 1x aten.add.Tensor(Tensor(torch.float32, torch.Size([32, 1024]), npu),Tensor(torch.float32, torch.Size([1, 1024]), npu)) -> (Tensor(torch.float32, torch.Size([32, 1024]), npu))
[2026-03-12 19:26:40] [RANK0] [INFO] Lowered 1x aten.sum.default(Tensor(torch.float32, torch.Size([32, 1024]), npu)) -> (Tensor(torch.float32, torch.Size([]), npu))
```

### 3.2 落盘文件说明

调试信息输出位于执行目录下的torch_compile_debug子目录，**带有 `autofused_` 前缀的目录**为 inductor-npu-ext 相关产物，其余均为 inductor 原生产物。

> 融合Kernel的命名格式为autofused_{原始节点类型}_{拓扑结构hash}

```text
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

> 如果只在整网执行下才可触发，可以修改该文件中的Ascend C源码，添加Ascend C print或者dump，观察UB中的数据以定位问题原因。执行该文件**会直接修改本地缓存**，下次模型执行生效。

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

```shell
python3 benchmark.py
```

不传入e2e时，执行该文件会执行AscendC代码编译到Profiling采集的完整流程。如果您了解AscendC，可以尝试手工优化其中的AscendC Kernel实现，并观察性能变化。通常用于指导AscIR的Codegen策略优化或快速验证修改方案。

```shell
python3 benchmark.py e2e
```

传入e2e参数时，执行该文件会执行AscIR表达到Profiling采集的完整流程。如果您优化了AscIR的Codegen实现或修复了Codegen的bug，可以通过该流程验证效果。

## 4. 持久化编译缓存

inductor-npu-ext会将融合Kernel编译结果进行缓存落盘。缓存在多进程、多线程、进程内的多个子进程以及同一进程内的相同pattern间共享，通过文件锁保证安全性。

缓存默认保存路径为`/tmp/.npu_kernels_{用户名}`。你也可以直接设置环境变量 `TORCHINDUCTOR_NPU_EXT_CACHE_DIR` 来指定inductor-npu-ext缓存目录。**注意**，请勿将缓存目录设置在多个OS共享的文件系统上，失去文件锁保护可能造成缓存损坏。

缓存目录下，每个融合kernel一个子目录，融合kernel子目录下，为以Codegen结果hash命名的二级子目录，用于处理不同软件版本间的Codegen实现差异。缓存匹配包含两部分：

- 融合结构的明文拓扑（不受节点名、符号名等冗余信息影响）
- Codegen结果的hash值

只有两者都匹配时才会命中缓存。

```text
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

## 5. 运行时异常说明

inductor-npu-ext 在运行过程中遇到错误时会抛出 `RuntimeError` 异常，异常信息中包含关键的诊断上下文。本节按异常类型分别说明含义、关注要点和后续处理方法。

### 5.1 RuntimeError: Failed to execute graph code

**异常堆栈示例：**

```bash
Traceback (most recent call last):
  File "test.py", line 11, in <module>
    out = test_add_sum(x, y)
  File ".../torch/_dynamo/eval_frame.py", line ..., in _fn
    ...
  File ".../torch/_inductor/graph.py", line ..., in compile_to_module
    ...
  File ".../inductor_npu_ext/codegen/_asc_codegen.py", line 58, in codegen_kernel_def
    raise RuntimeError(f"Failed to execute graph code:{graph_py_code.getvalue()} {e}") from e
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Failed to execute graph code:
from autofuse.pyautofuse import ascir
from autofuse.pyautofuse import Autofuser, AutofuserOptions
...
<原始错误信息，例如：TypeError: xxx() got an unexpected keyword argument 'yyy'>
```

**含义：** inductor-npu-ext 根据融合图结构生成了一段调用 CANN autofuse 接口的 Python 代码，该代码在执行时发生错误。

**关注要点：**

- 异常信息的**末尾部分**是原始 Python 错误（如 `TypeError`、`ValueError`、`KeyError` 等），这是定位问题的关键。
- 异常中间包含的大段代码是 inductor-npu-ext中 Codegen 生成的 AscIR 图代码。

**后续处理：**

1. 确认 CANN 和 inductor-npu-ext 版本是否匹配，参见 [软件安装](../README.md)。
2. 如果版本匹配仍然报错，请参考 [进一步帮助](#56-进一步帮助) 进行处理。

### 5.2 RuntimeError: Failed to build ascend kernel, trigger bug by following command

**异常堆栈示例：**

```bash
Traceback (most recent call last):
  File "test.py", line 11, in <module>
    out = test_add_sum(x, y)
  File ".../torch/_dynamo/eval_frame.py", line ..., in _fn
    ...
  File ".../inductor_npu_ext/compiler/_compiler.py", line 92, in build_ascend_lib
    raise RuntimeError(f"Failed to build ascend kernel, trigger bug by following command: ...")
RuntimeError: Failed to build ascend kernel, trigger bug by following command: 'python3 .../autofused_add_sum_31aba041b5d992c026cdfc574b77ec24/0b9cb30a2e017a8ba74972338f6e37fa/asc_kernel.py'
```

**含义：** inductor-npu-ext 生成的 Ascend C Kernel 代码编译失败。

**关注要点：**

- 异常信息中的 `trigger bug by following command` 部分给出了**完整的复现命令**，直接执行即可看到具体的编译错误输出。
- 执行后，关注编译器输出中的`error:`行，例如，`error: no matching function for call to 'AutofuseTiling'`，表明Tiling函数签名与CANN头文件不匹配。

**后续处理：**

1. 执行异常信息中的命令，查看编译器给出的具体错误。
2. 确认 CANN 版本是否满足 [软件安装](../README.md) 中的要求。
3. 如果版本匹配仍然报错，请参考 [进一步帮助](#56-进一步帮助) 进行处理。

### 5.3 RuntimeError: Failed to build wrapper, trigger bug by following command

**异常堆栈示例：**

```bash
Traceback (most recent call last):
  File "test.py", line 11, in <module>
    out = test_add_sum(x, y)
  File ".../torch/_dynamo/eval_frame.py", line ..., in _fn
    ...
  File ".../inductor_npu_ext/compiler/_compiler.py", line 101, in _build_cpp
    raise RuntimeError(f"Failed to build wrapper, trigger bug by following command: ...")
RuntimeError: Failed to build wrapper, trigger bug by following command: 'g++ -shared -std=c++17 -fPIC -Wall -O2 -o .../wrapper.so .../inductor_wrapper.cpp -I.../ascend/include -I.../torch/include -I.../torch_npu/include -L.../ascend/lib64 -lascendcl -lnnopbase -L.../torch_npu/lib -ltorch_npu'
```

**含义：** inductor-npu-ext 生成的 Host 端 C++ 代码（Wrapper）编译失败，通常与编译环境有关（缺少头文件、链接库或 GCC 版本不兼容）。

**关注要点：**

- 异常信息中的 `trigger bug by following command` 部分给出了**完整的 g++ 编译命令**，直接执行即可看到编译错误。
- 执行后，关注编译器输出中的`error:`行，例如，`error: 'xxx' was not declared in this scope`。

**后续处理：**

1. 执行异常信息中的 g++ 命令，查看具体的编译/链接错误。
2. 确认 CANN 环境变量已正确设置（`source /usr/local/Ascend/cann/set_env.sh`）、GCC 版本 >= 7.3.0、torch_npu 已正确安装。
3. 如果上述版本/环境变量确认完仍然报错，请参考 [进一步帮助](#56-进一步帮助) 进行处理。

### 5.4 RuntimeError: NPU kernel .npu_kernels_xxx init failed

**异常堆栈示例：**

```bash
Kernel load failed for autofused_add_sum_31aba041b5d992c026cdfc574b77ec24
Traceback (most recent call last):
  File "test.py", line 11, in <module>
    out = test_add_sum(x, y)
  File ".../torch/_dynamo/eval_frame.py", line ..., in _fn
    ...
  File ".../inductor_npu_ext/compiler/__init__.py", line 26, in __init__
    raise RuntimeError(f"NPU kernel {self.name} init failed")
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: NPU kernel .npu_kernels_xxx init failed
```

**含义：** 编译产物（`wrapper.so` 和 `kernel.so`）已生成，但在加载或初始化阶段失败。

**关注要点：** 异常中的 kernel 名称为缓存目录名（如 `.npu_kernels_xxx`），要定位具体是哪个融合 Kernel 出错，需要查看 **Trackback上方紧邻的输出**，其中包含 `autofused_xxx` 形式的 Kernel 名称。常见提示及含义如下：

| Trackback上方的提示信息 | 含义 |
| --- | --- |
| `Kernel load failed for autofused_xxx` | `kernel.so` 加载失败，通常为动态库损坏或依赖缺失 |
| `autofused_xxx kernel tiling func not found` | `kernel.so` 中未找到 `AutofuseTiling` 符号，通常为 `kernel.so` 与 `wrapper.so` 版本不一致 |
| `autofused_xxx kernel launch func not found` | `kernel.so` 中未找到 `AutofuseLaunch` 符号，原因同上 |
| `autofused_xxx kernel tiling failed` | 静态 Tiling 默认初始化执行失败，通常为 Tiling 逻辑错误 |

**后续处理：**

1. 根据提示信息中的 `autofused_xxx` 名称，找到执行目录下 `.npu_kernels_{用户名}/autofused_xxx` 子目录，删除后重新执行以排除缓存损坏问题。
2. 如果仍然报错，请参考 [进一步帮助](#56-进一步帮助) 进行处理。

### 5.5 RuntimeError: NPU kernel .npu_kernels_xxx execution failed(-1)

**异常堆栈示例：**

```bash
autofused_add_sum_31aba041b5d992c026cdfc574b77ec24 kernel tiling failed
Traceback (most recent call last):
  File "test.py", line 11, in <module>
    out = test_add_sum(x, y)
  File ".../torch/_dynamo/eval_frame.py", line ..., in _fn
    ...
  File ".../inductor_npu_ext/compiler/__init__.py", line 38, in __call__
    raise RuntimeError(f"NPU kernel {self.name} execution failed({result})")
RuntimeError: NPU kernel .npu_kernels_xxx execution failed(-1)
```

**含义：** Kernel 初始化已成功，但在执行下发过程中失败。

**关注要点：** 与 init failed 类似，需要查看 **Trackback上方紧邻的输出** 来定位具体出错的 Kernel。常见的提示及含义如下：

| Trackback上方的提示信息 | 含义 |
| --- | --- |
| `autofused_xxx kernel tiling failed` | 动态 Tiling 计算失败，通常为输入的动态 shape 值触发了 Tiling 逻辑中的异常路径 |
| `autofused_xxx kernel get stream failed` | 获取 NPU Stream 失败，通常为设备上下文异常 |
| `autofused_xxx kernel malloc workspace failed` | Workspace 内存申请失败，通常为设备内存不足 |
| `autofused_xxx kernel launch failed` | Kernel Launch 调用返回非零值，通常为 Ascend C Kernel 内部执行错误 |

**后续处理：**

1. 如果异常是异步上报的（报错位置与实际出错 Kernel 不一致），设置 `ASCEND_LAUNCH_BLOCKING=1` 重新执行，确保同步下发以定位首个报错的 Kernel。
2. 优先确认CANN和torch_npu版本是否满足 [软件安装](../README.md) 中的要求。
3. 其他可采取的措施：
   - 如果遇到 `tiling failed`：确认输入 shape 是否合法，动态 shape 场景下关注是否存在极端值（如 0 或超大值）。
   - 如果遇到 `malloc workspace failed`：检查设备内存是否充足，尝试减小 batch size 或释放其他显存占用。
4. 如无法自行解决，请参考 [进一步帮助](#56-进一步帮助) 进行处理。

### 5.6 进一步帮助

如果上述步骤仍无法解决问题，可到以下仓库提交issue：

- [AutoFusion 仓库](https://gitcode.com/cann/graph-autofusion/issues)：如遇到 5.2（Kernel 编译错误）以及 5.5（kernel执行报错）优先在该仓库提交
- [torchair 仓库](https://gitcode.com/Ascend/torchair/issues)

提交 issue 时请附上以下信息以便开发者快速定位：

- 完整的异常堆栈
- CANN、torch_npu、inductor-npu-ext 的版本信息
- 触发异常的模型或最小复现脚本（如有）
