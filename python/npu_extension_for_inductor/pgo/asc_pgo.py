# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import logging
from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import argparse
import subprocess
import queue
import torch
import torch_npu


device_pool = queue.Queue()
device_count = 0


def is_device_available(device_id: int) -> bool:
    """检查指定的 NPU 设备是否可用"""
    try:
        torch_npu.npu.set_device(device_id)
        _ = torch.tensor([1.0], device=f'npu:{device_id}')
        return True
    except Exception:
        return False


def initialize_device_pool() -> None:
    """初始化设备池"""
    global device_count
    for i in range(torch_npu.npu.device_count()):
        if is_device_available(i):
            device_count += 1
            device_pool.put(i)
        else:
            logging.warning("设备 %d 不可用，跳过", i)


def get_mspti_path() -> str:
    """获取 mspti 库的路径"""
    ascend_dir = os.path.dirname(os.getenv("ASCEND_OPP_PATH", "/usr/local/Ascend/latest/opp"))
    mspti_lib_path = os.path.join(ascend_dir, "tools/mspti/lib64/libmspti.so")
    return mspti_lib_path if os.path.exists(mspti_lib_path) else ""


def execute_pgo(pgo_path: str) -> None:
    """执行单个 pgo.py 文件"""
    device_id = device_pool.get()
    try:
        args = "python3"
        subprocess.run(
            [args, pgo_path, "-d", str(device_id)],
            check=True,
            env={
                    "LD_PRELOAD": get_mspti_path(),
                    **dict(os.environ)
                }
        )
    except Exception as e:
        logging.error("执行失败 %s: %s", pgo_path, e, exc_info=True)
    finally:
        device_pool.put(device_id)


def get_compile_concurrency(cli_max: int = 0) -> int:
    """获取编译并发数，优先使用命令行参数，否则按最大CPU数量并行，且不超过CPU数量"""
    cpu_count = os.cpu_count() or 1

    concurrency = cli_max if cli_max > 0 else cpu_count
    return concurrency


def compile_case(case_path: Path, mspti_path: str, run_pool: ThreadPoolExecutor) -> None:
    """编译单个 pgo.py 文件"""
    try:
        args = "python3"
        subprocess.run(
            [args, str(case_path), "compile"],
            check=True,
            env={
                    "LD_PRELOAD": mspti_path,
                    **dict(os.environ)
                }
        )
        run_pool.submit(run_case, case_path, mspti_path)
    except Exception as e:
        logging.error("编译失败 %s: %s", case_path, e, exc_info=True)


def run_case(case_path: Path, mspti_path: str) -> None:
    """运行编译后的 pgo.py 文件"""
    device_id = None
    try:
        device_id = device_pool.get()
        args = "python3"
        subprocess.run(
            [args, str(case_path), "run", "-d", str(device_id)],
            check=True,
            env={
                    "LD_PRELOAD": mspti_path,
                    **dict(os.environ)
                }
        )
    except Exception as e:
        logging.error("运行失败 %s: %s", case_path, e, exc_info=True)
    finally:
        if device_id is not None:
            device_pool.put(device_id)


def find_latest_run_dir(base_path: Path) -> Path:
    """查找 base_path 下以 'run_' 开头的最新目录"""
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        raise FileNotFoundError("未找到任何以 'run_' 开头的目录")
    # 按修改时间排序，取最新的目录
    latest_dir = max(run_dirs, key=lambda x: x.stat().st_mtime)
    return latest_dir


def collect_pgo_cases(inductor_dir: Path) -> List[Path]:
    """遍历 inductor_dir，收集所有存在的 pgo.py 路径"""
    case_paths = []
    for pgo_path in inductor_dir.rglob("pgo.py"):
        if pgo_path.is_file():
            case_paths.append(pgo_path)
    return case_paths


def run():
    parser = argparse.ArgumentParser(
        description="批量编译和运行 NPU PGO 优化脚本"
    )
    parser.add_argument(
        "-c", "--max-compile", type=int, default=None,
        help="最大编译并发数（默认四分之一CPU，最多64）"
    )
    parser.add_argument(
        "-r", "--max-run", type=int, default=8,
        help="最大调优并发数（默认8）"
    )
    args = parser.parse_args()
    max_compile = get_compile_concurrency(args.max_compile)
    max_run = args.max_run

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

    initialize_device_pool()

    case_paths = collect_pgo_cases(inductor_dir)

    mspti_path = get_mspti_path()

    with ThreadPoolExecutor(max_workers=max_run) as run_pool:
        with ThreadPoolExecutor(max_workers=max_compile) as compile_pool:
            futures = [
                compile_pool.submit(compile_case, path, mspti_path, run_pool)
                for path in case_paths
            ]
            wait(futures, return_when=ALL_COMPLETED)
        run_pool.shutdown(wait=True)

if __name__ == "__main__":
    run()
