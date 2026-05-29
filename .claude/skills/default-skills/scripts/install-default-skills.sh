#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e

# 脚本功能：安装项目必备的 skills
# 默认技能列表
DEFAULT_SKILLS=("gitcode-pr" "gitcode-issue" "api-doc-generator" "gitcode-pipeline")

# 超时时间（秒）
CONNECT_TIMEOUT=5
CLONE_TIMEOUT=20

# 仓库 URL
REPO_URL="https://gitcode.com/cann-agent/skills.git"

# 脚本所在目录的上上级目录为 .claude/skills/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILLS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REMOTE_DIR="$SKILLS_DIR/_remote"
GITIGNORE="$(cd "$SCRIPT_DIR/../../.." && pwd)/.gitignore"

# 创建远程 skills 子目录
mkdir -p "$REMOTE_DIR"

# 创建临时目录
if command -v mktemp &> /dev/null; then
    TEMP_DIR=$(mktemp -d)
else
    # Git Bash 或其他环境没有 mktemp，使用备用方案
    TEMP_DIR="/tmp/skills_install_$$"
    mkdir -p "$TEMP_DIR"
fi
trap 'rm -rf "$TEMP_DIR"' EXIT

# 检查网络连通性（快速检测是否可以访问仓库）
echo "Checking repository accessibility..."
check_connectivity() {
    if command -v curl &> /dev/null; then
        http_code=$(curl --connect-timeout $CONNECT_TIMEOUT --max-time $CONNECT_TIMEOUT -s -o /dev/null -w "%{http_code}" "https://gitcode.com" 2>/dev/null)
        if [ "$http_code" = "200" ] || [ "$http_code" = "301" ] || [ "$http_code" = "302" ]; then
            return 0
        fi
        return 1
    elif command -v wget &> /dev/null; then
        wget --timeout=$CONNECT_TIMEOUT -q --spider "https://gitcode.com" 2>/dev/null
        return $?
    else
        # 如果 curl 和 wget 都不存在，尝试使用 git 的低级探测
        GIT_HTTP_LOW_SPEED_LIMIT=1 GIT_HTTP_LOW_SPEED_TIME=$CONNECT_TIMEOUT \
            git ls-remote "$REPO_URL" HEAD &>/dev/null
        return $?
    fi
}

if ! check_connectivity; then
    echo "Error: Cannot access gitcode.com, please check network connectivity"
    exit 1
fi

# 克隆 skills 仓库
echo "Cloning skills repository..."
if command -v timeout &> /dev/null; then
    timeout $CLONE_TIMEOUT git clone --depth 1 "$REPO_URL" "$TEMP_DIR/skills" 2>&1
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clone skills repository"
        exit 1
    fi
else
    GIT_HTTP_LOW_SPEED_LIMIT=1000
    GIT_HTTP_LOW_SPEED_TIME=$CLONE_TIMEOUT
    export GIT_HTTP_LOW_SPEED_LIMIT GIT_HTTP_LOW_SPEED_TIME
    git clone --depth 1 "$REPO_URL" "$TEMP_DIR/skills" 2>&1
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clone skills repository"
        exit 1
    fi
fi

# 检查 skills 目录是否存在
if [ ! -d "$TEMP_DIR/skills/skills" ]; then
    echo "Error: skills directory not found in repository"
    exit 1
fi

# 拷贝技能到 .claude/skills/_remote/ 目录，并在 skills 根目录创建符号链接
echo "Installing skills..."
for skill in "${DEFAULT_SKILLS[@]}"; do
    if [ -d "$TEMP_DIR/skills/skills/$skill" ]; then
        cp -r "$TEMP_DIR/skills/skills/$skill" "$REMOTE_DIR/"
        # 创建符号链接使 Claude Code 能发现 _remote 下的 skill（仅扫描一级目录）
        rm -rf "$SKILLS_DIR/$skill"
        ln -sfn "_remote/$skill" "$SKILLS_DIR/$skill"
        echo "Installed skill: $skill"
    else
        echo "Warning: Skill '$skill' not found in repository"
    fi
done

# 更新 .gitignore：确保符号链接被忽略（幂等，已有条目不会重复添加）
if [ -f "$GITIGNORE" ]; then
    for skill in "${DEFAULT_SKILLS[@]}"; do
        ENTRY=".claude/skills/$skill"
        if ! grep -qxF "$ENTRY" "$GITIGNORE"; then
            echo "$ENTRY" >> "$GITIGNORE"
        fi
    done
fi

echo "All skills installed successfully."
