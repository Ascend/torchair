# AGENTS.md

## 项目概述

TorchAir（Torch Ascend Intermediate Representation）是一个 PyTorch 的昇腾　NPU 图模式扩展库，它基于 PyTorch Dynamo 捕获 FX 计算图，并将其转换为 GE（Graph Engine）计算图进行全图编译执行，或通过 ACL（Ascend Computing Language）Graph 以捕获方式执行，从而实现 Pytorch 模型在昇腾 NPU 上的零改动推理加速．

- 仓库地址：https://gitcode.com/ascend/torchair
- 支持的 PyTorch 版本：2.6.0+（master 主线），2.1.0–2.9.0（已发布版本）
- Python 版本：3.9–3.12

## 构建命令

构建前需执行一次 configure，用于检测 Python/torch 环境及可选的 Ascend SDK 路径。

```bash
# 交互式配置
bash ./configure

# 非交互式配置
TARGET_PYTHON_PATH=/usr/bin/python3 NO_ASCEND_SDK=1 bash ./configure
# 指定 SDK：TARGET_PYTHON_PATH=/usr/bin/python3 ASCEND_SDK_PATH=~/Ascend/ascend-toolkit/latest/ bash ./configure
```

编译与安装：

```bash
mkdir build && cd build
cmake ..
make torchair -j8          # 编译生成 build/dist/dist/torchair-0.1-py3-none-any.whl
make install_torchair      # 安装到 configure 时指定的 Python 环境
```

一键构建脚本：

```bash
bash build.sh -c                   # 仅编译（CI 模式）
bash build.sh -i                   # 编译 + 安装（CI 模式）
bash build.sh -u                   # 运行单元测试（需设置 ASCEND_HOME_PATH 或 ASCEND_CUSTOM_PATH）
bash build.sh -s                   # 运行系统测试
bash build_and_install.sh [dst]    # 快速编译 + 安装（可指定安装目录）
```

## 测试

- **C++ 单元测试**：gtest 二进制文件构建于 `tests/build/torchair/ut/torchair_ut_main`，通过 `bash build.sh -u` 运行。
- **Python 单元测试**：`python -m unittest discover -s experimental/_inductor_npu_ext/tests -p "*.py"`（`build.sh -u` 会自动执行）。
- **系统测试**：`tests/st/` 目录下的 Python 脚本（如 `torchair_st.py`、`torchair_aclgraph_st.py`），通过 `bash build.sh -s` 运行。
- **运行单个 C++ 测试**：编译测试目标后，运行 gtest 二进制时加 `--gtest_filter=TestSuite.TestCase`。

在没有 NPU 硬件的环境下测试时，执行 `source tools/env.sh` 设置 `LD_LIBRARY_PATH`，使其包含 stub .so 文件和 SDK 库。

## 架构

### 编译流程

1. 用户调用 `torch.compile(model, backend=torchair.get_npu_backend(compiler_config=config))`
2. PyTorch Dynamo 将模型捕获为 FX 图
3. `npu_fx_compiler.py` 编排编译流程：FX 图 → GE 图转换 → GE 编译 → NPU 执行
4. 编译后的 GE 图被缓存，通过 NPU graph executor 执行

### C++ 核心（`torchair/`）

- `core/` — 运行时工具、日志、环境配置、错误处理
- `concrete_graph/` — 计算图的 C++ 表示（FX 与 GE 之间的"concrete graph"中间表示）
- `abi_compat_ge_apis/` — GE API 的 ABI 兼容封装，用于解耦具体 GE 版本
- `npu_graph_executor/` — NPU 图执行器：管理已编译图的生命周期、会话创建及设备端执行
- `npu_utils/` — 向 Python 暴露的 NPU 工具函数（设备查询、格式转换等）
- `cann_ir_ability/` — CANN IR（中间表示）分析与变换能力
- `llm_datadist/` — LLM 负载的数据分布工具

### Python 包（`python/torchair/`）

- `npu_fx_compiler.py` — 入口：`get_compiler()` / `get_npu_backend()`，编排 FX→GE 编译流水线
- `npu_export.py` — `dynamo_export()`：将模型导出为 AIR 格式
- `configs/compiler_config.py` — `CompilerConfig` 数据类，控制全部编译选项（融合、缓存、日志等）
- `_ge_concrete_graph/` — FX→GE 转换器注册表及逐算子转换逻辑。`fx2ge_converter.py` 是分发表；`ge_converter.py` 负责具体转换
- `_acl_concrete_graph/` — 基于 ACL 的图表示（GE 路径的替代方案）
- `ge/` — GE 图的 Python 绑定及自定义算子支持
- `ge_concrete_graph/` — 高层 GE concrete graph 工具
- `patterns/` — 模式匹配与替换 Pass，用于图优化（融合、简化）
- `ops/` — 注册到 NPU 的自定义 torch 算子
- `experimental/` — 实验性功能（推理优化、inductor 扩展）
- `inference/` — 稳定的推理工具（权重格式转换等）
- `scope/` — 算子归因的作用域追踪
- `llm_datadist/` — LLM 数据分布的 Python 侧支持

### 代码生成（`codegen/`）

- `codegen.cpp` — 生成 GE 原始算子的 Python 封装（`auto_generated_ge_raw_ops.py`）及自定义算子。编译为 `code_generator` 和 `code_generator_single` 两个可执行文件。
- `custom_op/` — 自定义算子代码生成模板

### CMake 模块（`cmake/`）

每个模块（`acl`、`aoe`、`graph_engine`、`hccl`、`metadef`、`pytorch`、`torch_npu`）负责定位并导入对应依赖的头文件和库。configure 步骤将路径写入 `tools/` 目录（TORCH_INSTALLED_PATH、PYTHON_BIN_PATH、COMPILE_FLAGS、TORCH_VERSION）。

### 测试（`tests/`）

- `ut/testcases/` — C++ 单元测试（gtest），覆盖 concrete_graph、core 等模块
- `st/` — 系统/集成测试（Python），每个 `torchair_*_st.py` 测试一个端到端场景
- `tests/CMakeLists.txt` 构建 stub 库（`tests/build/stubs/` 下的 `.so`），使 C++ 测试无需真实 NPU 硬件即可链接

## 核心 API

- `torchair.get_npu_backend(compiler_config=config)` — 返回供 `torch.compile()` 使用的 Dynamo 后端
- `torchair.CompilerConfig()` — 编译选项配置对象
- `torchair.dynamo_export(model, *args)` — 将模型导出为 AIR 格式
- `torchair.register_fx_node_ge_converter(aten_op)` — 为指定 aten 算子注册自定义 FX→GE 转换器
- `torchair.register_replacement(pattern, replacement)` — 注册基于模式的图优化替换规则

## 代码检查

pre-commit 钩子配置于 `.pre-commit-config.yaml`：trailing-whitespace、end-of-file-fixer、check-yaml、codespell（仅非 Python/C++ 文件）、bandit（Python 安全检查）、typos。注意：ruff、pylint、clang-format 已在配置中但当前被注释。

```bash
pre-commit run --all-files
```
