import os
import warnings
import subprocess
from pathlib import Path
import datetime

import torch
import torch.distributed as dist
from torchair.core import _torchair
from torchair.core.utils import logger


_uninstall_paths = []


def compile_static_kernel(fx_func, *args, build_dir=None, **kwargs):
    if not _is_single_card() and "LOCAL_WORLD_SIZE" not in os.environ:
        warnings.warn(
        "Environment variables 'LOCAL_WORLD_SIZE' is not set in a multi-card context. "
        "As a result, the static kernel feature will be disabled. "
        "To resolve this, please either launch the process via torchrun,"
        "or manually configure the 'LOCAL_WORLD_SIZE' environment variables. "
        )
        return

    warnings.warn("Starting static kernel compilation")
    result_root = safe_resolve_output_dir(build_dir)

    # 1.执行单算子，用于生成算子信息json
    _torchair.AclopStartDumpArgs(1, str(result_root))
    try:
        if isinstance(fx_func, torch.fx.GraphModule):
            torch.fx.Interpreter(fx_func).run(*args, **kwargs)
        else:
            fx_func(*args, **kwargs)
        import torch_npu
        torch_npu.npu.current_stream().synchronize()
    finally:
        _torchair.AclopStopDumpArgs(1)
    logger.debug("static kernel run eager success")

    debug_dirs = [d for d in result_root.iterdir() if d.is_dir() and d.name.endswith("_debug")]
    if not debug_dirs:
        logger.debug("Can not find json of ops, do not execute op_compiler")
        return

    debug_dir = max(debug_dirs, key=lambda d: d.stat().st_mtime)
    json_files = list(debug_dir.glob("*.json"))
    if not json_files:
        logger.debug(f"No JSON files in {debug_dir}, skip op_compiler")
        return

    # 2.开始静态编译
    cmd = [
        "op_compiler",
        "-p", str(debug_dir),
        "-v", _torchair.GetSocName(),
        "-l", "info",
        "-j", "4",
        "-o", str(result_root),
    ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(f"execute op_compiler, msg: {res.stdout}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"execute op_compiler error, msg: {e.stderr}")
        return

    # 3.安装静态kernel run包
    install_and_sync_run_pkgs(result_root)


def _is_single_card() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True

    try:
        return dist.get_world_size() <= 1
    except Exception:
        return True


def _install_run_packages(result_root: Path, rank: int = None):
    run_pkgs = list(result_root.glob("*.run"))
    prefix = f"Rank {rank}: " if rank is not None else ""
    if not run_pkgs:
        logger.debug(f"{prefix}no static kernel run packages to install")
        return

    for run_pkg in run_pkgs:
        filename = run_pkg.name
        try:
            res = subprocess.run([str(run_pkg)], check=True, capture_output=True, text=True)
            logger.debug(f"{prefix}{filename} install success, msg: {res.stdout}")
            save_uninstall_info(filename[:-4])
        except subprocess.CalledProcessError as e:
            logger.warning(f"{prefix}{filename} install failed, msg: {e.stderr}")


def _reselect_static_kernel(rank: int = None):
    prefix = f"Rank {rank}: " if rank is not None else ""
    try:
        import torch_npu
        torch_npu.npu._aclnn_reselect_static_kernel()
        logger.debug(f"{prefix}reselect_static_kernel executed successfully")
    except Exception as e:
        logger.warning(f"{prefix}reselect_static_kernel failed: {e}")


# Example calculations:
# Nodes: 3 (Node A, Node B, Node C)
# GPUs per node: 4
# LOCAL_WORLD_SIZE = 4
# +---------------------+-------------+-------------+-------------------+----------------------+
# | Example Process     | node_idx    | global_rank | start = node_idx * | group_ranks         |
# |                     |             |             | local_world_size   |                     |
# +---------------------+-------------+-------------+--------------------+---------------------+
# | Node A GPU 2        | 0           | 2           | 0 * 4 = 0          | [0, 1, 2, 3]        |
# | Node B GPU 1        | 1           | 5           | 1 * 4 = 4          | [4, 5, 6, 7]        |
# | Node C GPU 3        | 2           | 11          | 2 * 4 = 8         | [8, 9, 10, 11]      |
# +---------------------+-------------+-------------+--------------------+---------------------+
def _get_local_gloo_group():
    try:
        local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    except KeyError as e:
        raise RuntimeError(f"missing environment variable {e.args[0]}, cannot create local Gloo process group") from e
    except ValueError as e:
        raise RuntimeError("environment variables LOCAL_WORLD_SIZE must be integers") from e
    except Exception as e:
        logger.warning(f"get LOCAL_WORLD_SIZE failed: {e}")

    if local_world_size <= 0:
        raise RuntimeError("LOCAL_WORLD_SIZE must be > 0")

    world_size = dist.get_world_size()
    if world_size % local_world_size != 0:
        raise RuntimeError(f"world_size ({world_size}) is not divisible by LOCAL_WORLD_SIZE ({local_world_size}); ")

    global_rank = dist.get_rank()
    node_count = world_size // local_world_size
    local_gloo_groups = [None] * node_count

    for node_idx in range(node_count):
        start = node_idx * local_world_size
        ranks = list(range(start, start + local_world_size))
        local_gloo_groups[node_idx] = dist.new_group(ranks=ranks, backend="gloo")

    return local_gloo_groups[global_rank // local_world_size]


def install_and_sync_run_pkgs(result_root: Path):
    # 1. 检查是多卡还是单卡
    if _is_single_card():
        logger.debug("single card")
        _install_run_packages(result_root)
        _reselect_static_kernel()
        return

    # 2. 进入多卡流程
    rank = dist.get_rank()
    gloo_group = _get_local_gloo_group()

    logger.debug(f"Rank {rank}: start installing static kernel .run packages")
    _install_run_packages(result_root, rank)

    # 3. barrier 等待所有安装完
    logger.debug(f"Rank {rank}: barrier after install (Gloo)")
    dist.barrier(group=gloo_group)

    # 4. 所有 rank 执行 reselect
    logger.debug(f"Rank {rank}: start reselecting static kernel")
    _reselect_static_kernel(rank)

    # 5. barrier 等待所有 reselect 完
    logger.debug(f"Rank {rank}: barrier after reselect (Gloo)")
    dist.barrier(group=gloo_group)


def save_uninstall_info(filename: str):
    global _uninstall_paths
    latest = Path(os.environ["ASCEND_HOME_PATH"])
    root = latest.parent
    pattern = f"*/opp/static_kernel/ai_core/{filename}/uninstall.sh"
    match = next(root.glob(pattern), None)
    if match is None:
        logger.debug(f"can not find uninstall path, pattern: {pattern}")
    else:
        _uninstall_paths.append(str(match))
        warnings.warn(
            "Warning: If the process exits abnormally, "
            f"you must manually uninstall the static kernel package by executing: {match}"
        )


def uninstall_static_kernel():
    global _uninstall_paths
    if not _uninstall_paths:
        logger.debug(f"no static kernel uninstall paths recorded, skip uninstall static kernels")
        return

    for script_path in _uninstall_paths:
        try:
            result = subprocess.run(
                [script_path], check=True, capture_output=True, text=True
            )
            logger.debug(f"{script_path} uninstall success")
        except subprocess.CalledProcessError as e:
            logger.error(f"{script_path} uninstall failed, msg:\n{e.stderr}")

    _uninstall_paths.clear()


def safe_resolve_output_dir(build_dir: str):
    base_dir = Path.cwd().resolve()
    if build_dir is not None:
        if "\x00" in build_dir:
            raise ValueError("build_dir contains null byte")

        candidate = Path(build_dir)
        if ".." in candidate.parts:
            raise ValueError("build_dir must not contain '..'")

        script_dir = candidate if candidate.is_absolute() else base_dir / candidate

        cur = Path(script_dir.anchor)
        for part in script_dir.parts[1:]:
            cur = cur / part
            if cur.exists() and cur.is_symlink():
                raise ValueError(f"symlink detected in path: {cur}")

        try:
            script_dir = script_dir.resolve(strict=False)
        except Exception as e:
            raise ValueError(f"cannot resolve path {script_dir}: {e}")
    else:
        script_dir = base_dir  # 在同目录生成临时dump的文件夹，用于保存生成的算子信息json

    base_output_dir = script_dir / "aclnn_static_shape_kernel_outputs"
    try:
        base_output_dir.mkdir(exist_ok=True)
    except (PermissionError, OSError) as e:
        raise RuntimeError(f"failed to create base output directory {base_output_dir}: {e}") from e

    timestamp = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}_{os.getpid()}"
    result_root = base_output_dir / f"{timestamp}_outputs"

    try:
        result_root.mkdir(exist_ok=True)
    except (PermissionError, OSError) as e:
        raise RuntimeError(f"failed to create output directory {result_root}: {e}") from e

    return result_root
