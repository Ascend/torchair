import os
import subprocess
import datetime
from enum import IntEnum
from pathlib import Path
import warnings

import torch
from torchair.core import _torchair
from torchair.core.utils import logger


_uninstall_path = None


def compile_static_kernel(fx_graph, *args, build_dir=None, **kwargs):
    warnings.warn("Starting static kernel compilation")

    result_root = safe_resolve_output_dir(build_dir)

    # 执行单算子，用于生成算子信息json
    import acl
    acl.op.start_dump_args(1, str(result_root))
    try:
        torch.fx.Interpreter(fx_graph).run(*args, **kwargs)
    finally:
        acl.op.stop_dump_args(1)
    logger.debug("static kernel run eager success")

    debug_dirs = [d for d in result_root.iterdir()
                  if d.is_dir() and d.name.endswith("_debug")]
    if not debug_dirs:
        logger.debug("Can not find json of ops, do not excute op_compiler")
        return

    debug_dir = max(debug_dirs, key=lambda d: d.stat().st_mtime)
    json_files = list(debug_dir.glob("*.json"))
    if not json_files:
        logger.debug(f"No JSON files in {debug_dir}, skip op_compiler")
        return

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
        logger.debug(f"excute op_compiler, msg: {res.stdout}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"excute op_compiler error, msg: {e.stderr}")
        return

    for run_pkg in result_root.glob("*.run"):
        filepath = run_pkg
        filename = run_pkg.name
        try:
            result = subprocess.run(
                [str(filepath)],
                check=True,
                capture_output=True,
                text=True
            )
            logger.debug(f"{filename} static kernel run pkg install success, msg: {result.stdout}")
            save_uninstall_info(filename[:-4])
            import torch_npu
            torch_npu.npu._aclnn_reselect_static_kernel()
        except subprocess.CalledProcessError as e:
            logger.warning(f"{filename} static kernel run pkg install failed, msg: {e.stderr}")


def save_uninstall_info(filename: str):
    global _uninstall_path
    latest = Path(os.environ["ASCEND_HOME_PATH"])
    root = latest.parent
    pattern = f"*/opp/static_kernel/ai_core/{filename}/uninstall.sh"
    match = next(root.glob(pattern), None)
    if match is None:
        _uninstall_path = None
        logger.debug(f"can not find uninstall path, pattern: {pattern}")
    else:
        _uninstall_path = str(match)


def uninstall_static_kernel():
    global _uninstall_path
    if not _uninstall_path:
        logger.debug(f"uninstall_path is none, skip uninstall static kernel")
        return

    try:
        result = subprocess.run(
            [_uninstall_path],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.debug(f"{_uninstall_path} uninstall success")
    except subprocess.CalledProcessError as e:
        logger.error(f"{_uninstall_path} uninstall failed, msg: \n{e.stderr}")
    finally:
        _uninstall_path = None


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

    base_output_dir = script_dir / "kernel_aot_optimization_outputs"
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
