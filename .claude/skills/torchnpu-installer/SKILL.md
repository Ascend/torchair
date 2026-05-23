---
name: torchnpu-installer
description: 在 conda 环境中安装 torch_npu 包。根据 conda 环境的 torch、python 版本及日期自动构建下载链接，从华为镜像站下载并安装对应的 torch_npu 包。
---

# torch_npu 包自动安装

自动从华为镜像站下载并安装 torch_npu 包，包含 conda 环境检测、版本匹配、下载、解压和安装流程。

## 快速开始

```bash
# 自动使用今天日期，默认目录 ./torch_npu_pkg
/torchnpu-installer myenv

# 指定日期
/torchnpu-installer myenv 20260520

# 指定日期和目录
/torchnpu-installer myenv 20260520 /path/to/pkgdir
```

## 完整参数

```bash
/torchnpu-installer <conda-name> <date> <pkg-dir>
```

- **conda-name**（必填）：需要安装的 conda 环境名
- **date**（可选）：构建日期，格式 `YYYYMMDD`，默认使用今天
- **pkg-dir**（可选）：下载并解压路径，默认 `./torch_npu_pkg`

## 安装流程

### 1. 参数解析

- 检查用户是否输入 `conda-name`，如果没有，提示用户输入
- `date` 格式 `YYYYMMDD`，默认使用当天日期（由 `date +%Y%m%d` 生成）
- `pkg-dir` 指定下载和解压目录，默认 `./torch_npu_pkg`，如果目录已存在则直接使用

### 2. 获取 conda 环境信息

激活目标 conda 环境，获取以下信息：

```bash
conda run -n <conda-name> python -c "import torch; print(torch.__version__)"
conda run -n <conda-name> python -c "import sys; print(f'py{sys.version_info.major}{sys.version_info.minor}')"
```

### 3. 替换下载链接

原始链接：

```bash
https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.12.0/20260520.1/pytorch_v2.12.0_py313.tar.gz
```

需要替换的部分：

- `v2.12.0` → torch 版本号（如 `v2.5.0`、`v2.9.0`，注意带 v 前缀）
- `py313` → python 标识符（如 `py313`、`py312`、`py311`）
- `20260520` → 用户输入的日期

### 4. 下载并解压

```bash
# 创建并进入 pkg-dir 目录（默认 ./torch_npu_pkg）
mkdir -p <pkg-dir>
cd <pkg-dir>

# 下载包
wget -O "pytorch.tar.gz" "<替换后的链接>"

# 解压到当前目录
tar -xzf pytorch.tar.gz
```

解压后，whl 文件会直接存在于 `<pkg-dir>` 目录下。

### 5. 检测架构并安装

检测当前系统架构：

```bash
arch=$(uname -m)
# x86_64/i686/i386 → x86
# aarch64/armv8* → arm
```

根据架构选择对应的 torch_npu 包进行安装：

```bash
conda run -n <conda-name> pip install --force-reinstall ./<arm或x86路径>/torch_npu*.whl
```

## 故障排查

### 找不到 conda 环境

```bash
conda env list
```

### 网络问题

```bash
curl -I https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/
```

### 架构不匹配

安装前确认目标 conda 环境是 x86 还是 arm 架构，torch_npu 包需要与 python 版本和 torch 版本匹配。
