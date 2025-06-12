# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess


def execute_pgo(pgo_path):
    """执行单个 pgo.py 文件"""
    try:
        args = "python3"
        subprocess.run([args, pgo_path], check=True)
    except Exception as e:
        logging.error("执行失败 %s: %s", pgo_path, e, exc_info=True)


def find_latest_run_dir(base_path: Path) -> Path:
    """查找 base_path 下以 'run_' 开头的最新目录"""
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        raise FileNotFoundError("未找到任何以 'run_' 开头的目录")
    # 按修改时间排序，取最新的目录
    latest_dir = max(run_dirs, key=lambda x: x.stat().st_mtime)
    return latest_dir


def run():
    base_path = Path("./torch_compile_debug")

    try:
        latest_run_dir = find_latest_run_dir(base_path)
        logging.info("将处理最新目录: %s", latest_run_dir)
    except FileNotFoundError as e:
        logging.error("%s", e, exc_info=True)
        return

    inductor_dir = latest_run_dir / "torchinductor"
    if not inductor_dir.exists():
        logging.error("未找到 'torchinductor' 子目录: %s", inductor_dir)
        return

    with ThreadPoolExecutor(max_workers=4) as executor:
        for model_dir in inductor_dir.iterdir():
            if not model_dir.is_dir():
                continue

            sub_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
            if len(sub_dirs) != 1:
                logging.info(f"模型目录 %s 下未找到唯一算子目录（找到 %d 个）", model_dir, len(sub_dirs))
                continue

            op_dir = sub_dirs[0]

            pgo_path = op_dir / "pgo.py"
            if pgo_path.exists():
                executor.submit(execute_pgo, pgo_path)
            else:
                logging.info("未找到 pgo.py: %s", pgo_path)

if __name__ == "__main__":
    run()
