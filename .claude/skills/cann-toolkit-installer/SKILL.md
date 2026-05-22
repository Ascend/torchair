---
name: cann-toolkit-installer
description: 自动下载并安装华为 CANN Toolkit 开发套件。当用户需要安装、更新或重新安装 CANN Toolkit 时使用此 skill，适用于：新环境初始化、版本升级。自动获取最新构建版本，静默安装并输出详细摘要。用户可能的指令是更新toolkit，更新run包等
---

# CANN Toolkit 自动安装

自动从华为镜像站下载并安装社区版 CANN Toolkit 包，包含完整的下载、验证、安装和配置流程。

## 快速开始

提示大约需要10分钟，消耗1～2K tokens

```bash
# 自动检测架构并安装（推荐）
/cann-toolkit-installer

# 指定架构（覆盖自动检测）
/cann-toolkit-installer aarch64

# 完整参数
/cann-toolkit-installer <arch> <version> <install-path>
```

## 安装流程

### 1. 参数解析与架构检测

- **架构自动检测**：使用 `uname -m` 自动检测系统架构
  - `x86_64|i686|i386` → `x86_64`
  - `aarch64|armv8*|armv7l` → `aarch64`
  - 用户可通过 `--arch` 参数手动覆盖
- `version`: CANN 版本号（默认 9.0.0）
- `install-path`: 安装路径（可选，默认由安装程序选择）

### 2. 获取最新构建

**获取最新构建日期**：使用 curl 解析镜像站目录获取最新的构建日期（构建日期会动态变化）

```bash
# 获取最新的构建日期目录
latest_date=$(curl -sL "https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/" | grep -oE 'href="[0-9]+' | sed 's/href="//' | sort -r | head -1)
```

目录结构：`master/{build_date}/Ascend-cann-toolkit_{version}_linux-{arch}.run`

**注意**：

- 构建日期（如 `20260401000324385`）会动态变化，需要实时获取
- 完整 URL 格式：`https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/{build_date}/`

### 3. 下载 Toolkit 包

```bash
# 使用 wget 下载（支持进度显示）
wget -O "Ascend-cann-toolkit_${version}_linux-${arch}.run" \
  "https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/${latest_date}/Ascend-cann-toolkit_${version}_linux-${arch}.run" \
  --progress=bar:force
```

- 文件大小：约 1.2 GB
- 下载位置：当前工作目录或安装目录

### 4. 验证文件完整性

```bash
# 计算 SHA256 校验和
sha256sum Ascend-cann-toolkit_${version}_linux-${arch}.run

# 检查文件类型
file Ascend-cann-toolkit_${version}_linux-${arch}.run
```

### 5. 静默安装

```bash
# 执行静默安装
bash Ascend-cann-toolkit_${version}_linux-${arch}.run --install --force -q 2>&1 | tee install.log
```

参数说明：

- `--install`: 安装模式
- `--force`: 强制更新，使用上一次的目录
- `-q`: 静默模式（减少输出）
- `2>&1 | tee install.log`: 同时输出到终端和日志文件

### 6. 结果验证

- 检查安装日志中的 ERROR 信息
- 提取安装摘要，也就是安装包最后的 Summary 信息
- 检查环境变量配置状态

## 不需要显示已安装组件

## 故障排查

### 安装失败

**网络问题：**

```bash
# 测试镜像站连接
curl -I https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/
```

**权限问题：**

- 检查安装路径写权限
- 某些操作可能需要 sudo

**磁盘空间：**

```bash
# 检查可用空间（至少需要 2GB）
df -h
```

**依赖缺失：**

- bash >= 5.1.16
- python3 >= 3.9.x
- cmake >= 3.16.0

不需要验证安装，不需要读取默认的${HOME}/var/log/ascend_seclog/ascend_toolkit_install.log，因为里面可能有之前的错误日志，会干扰结论

## 注意事项

1. **网络要求**：需要访问华为镜像站（中国境内网络）
2. **安装时间**：5-10 分钟，取决于网络速度
3. **残留文件**：如果安装目录存在旧文件，会显示警告但仍继续
4. 安装目录如果是一个日期格式的，不表示上一个安装版本也是个日期的，因为可能是一个更新的版本安装在了以前的旧目录中
