import os
from enum import Enum
import warnings
import subprocess
from typing import Union
from pathlib import Path
import datetime
import hashlib
import shutil

import torch
import torch.distributed as dist

from npugraph_ex.core.utils import logger
from npugraph_ex._utils.graph_utils import debug_time

_installed_run_pkgs = set()
_uninstall_paths = []
_local_gloo_groups = None
_compiled_op_set = set()


class RunPackageStatus(Enum):
    NEED_SKIP = "need_skip"
    NEED_COMPILE = "need_compile"
    INSTALLED = "installed"


def compile_static_kernel(fx_func, *args, use_cache_compile=None, cached_cann_version=None, compile_cache_dir=None,
                          cached_deterministic=None, super_kernel_optimize=None, build_dir=None, **kwargs):
    if not _is_multicard_env_valid():
        return

    if use_cache_compile == "1":
        _install_or_compile_static_kernel(fx_func, *args, cached_cann_version=cached_cann_version,
                                          compile_cache_dir=compile_cache_dir,
                                          cached_deterministic=cached_deterministic,
                                          super_kernel_optimize=super_kernel_optimize, **kwargs)
    else:
        warn_msg = ("The current version now supports caching run packages from static kernel compilation, "
                    "it is recommended to delete the compilation cache and recompile "
                    "when you use `torch.npu.npugraph_ex.inference.cache_compile` and do not use debug.run_eagerly=True.")
        warnings.warn(warn_msg)
        warnings.filterwarnings("ignore", message=warn_msg)
        _compile_static_kernel(fx_func, *args, build_dir=build_dir, super_kernel_optimize=super_kernel_optimize, **kwargs)


def _install_or_compile_static_kernel(fx_func, *args, cached_cann_version=None, compile_cache_dir=None,
                                      cached_deterministic=None, super_kernel_optimize=None, **kwargs):
    if _is_single_card():
        _install_or_compile_for_single(fx_func, *args, cached_cann_version=cached_cann_version,
                                       compile_cache_dir=compile_cache_dir,
                                       cached_deterministic=cached_deterministic, super_kernel_optimize=super_kernel_optimize, **kwargs)
    else:
        _install_or_compile_for_multi(fx_func, *args, cached_cann_version=cached_cann_version,
                                      compile_cache_dir=compile_cache_dir,
                                      cached_deterministic=cached_deterministic, super_kernel_optimize=super_kernel_optimize, **kwargs)


def _install_or_compile_for_single(fx_func, *args, cached_cann_version=None, compile_cache_dir=None,
                                   cached_deterministic=None, super_kernel_optimize=None, **kwargs):
    if not _check_for_load_run_pkg(cached_cann_version, compile_cache_dir, cached_deterministic):
        return
    run_package_status = _find_and_install_run_pkgs(compile_cache_dir)
    if run_package_status == RunPackageStatus.INSTALLED:
        _reselect_static_kernel()
    elif run_package_status == RunPackageStatus.NEED_COMPILE:
        _compile_static_kernel_for_single_card(fx_func, *args, is_cache_compile=True, build_dir=compile_cache_dir,
                                               super_kernel_optimize=super_kernel_optimize, **kwargs)


def _install_or_compile_for_multi(fx_func, *args, cached_cann_version=None, compile_cache_dir=None,
                                  cached_deterministic=None, super_kernel_optimize=None, **kwargs):
    # 多卡流程
    gloo_group = _get_local_gloo_group()
    rank = dist.get_rank()

    # 1. local rank 0查找run包并执行install
    local_rank = dist.get_rank(group=gloo_group)
    run_package_status = None
    if local_rank == 0:
        logger.debug(f"Rank {rank}: start to load and install run packages.")
        check_result = _check_for_load_run_pkg(cached_cann_version, compile_cache_dir, cached_deterministic)
        if check_result:
            run_package_status = _find_and_install_run_pkgs(compile_cache_dir, rank)
        else:
            run_package_status = RunPackageStatus.NEED_SKIP

    # 2.广播local rank 0的执行结果
    run_package_status_list = [run_package_status]
    dist.broadcast_object_list(run_package_status_list, src=_get_group_root_rank(rank), group=gloo_group)
    received_run_package_status = run_package_status_list[0]
    logger.debug(f"Rank {rank}: the received_run_package_status is {received_run_package_status}")

    # 3.如果是install，则所有卡执行reselect并销毁通信组，否则进行编译
    if received_run_package_status != RunPackageStatus.NEED_COMPILE:
        if received_run_package_status == RunPackageStatus.INSTALLED:
            _reselect_static_kernel(rank)
        logger.debug(f"Rank {rank}: barrier before destroy (Gloo)")
        dist.barrier(group=gloo_group)
        try:
            destroy_local_gloo_groups()
        except Exception as e:
            logger.warning(f"failed to destroy local gloo groups after install_and_sync_run_pkgs: {e}")
    else:
        _compile_static_kernel_for_multi_card(fx_func, *args, gloo_group=gloo_group, is_cache_compile=True,
                                              build_dir=compile_cache_dir, super_kernel_optimize=super_kernel_optimize, **kwargs)


def _compile_static_kernel(fx_func, *args, is_cache_compile=False, build_dir=None, super_kernel_optimize=None, **kwargs):
    if _is_single_card():
        _compile_static_kernel_for_single_card(fx_func, *args, is_cache_compile=is_cache_compile,
                                               build_dir=build_dir, super_kernel_optimize=super_kernel_optimize, **kwargs)
    else:
        _compile_static_kernel_for_multi_card(fx_func, *args, is_cache_compile=is_cache_compile,
                                              build_dir=build_dir, super_kernel_optimize=super_kernel_optimize, **kwargs)


def _compile_static_kernel_for_single_card(fx_func, *args, is_cache_compile=False, build_dir=None, super_kernel_optimize=None, **kwargs):
    result_root = safe_resolve_output_dir(build_dir)
    warnings.warn(f"Starting static kernel compilation, the build directory is {result_root}")

    # 1.执行单算子，用于生成算子信息json
    _fx_func_run(args, fx_func, kwargs, result_root)
    # 获取待编译的json目录
    chosen_dir = _get_dumpjson_dir_for_opcompile(result_root, is_cache_compile)
    if chosen_dir is None:
        return

    # 2.开始静态编译
    compile_result = static_compile(result_root, chosen_dir, super_kernel_optimize=super_kernel_optimize)
    if not compile_result:
        return

    # 3.安装静态kernel run包
    _install_run_packages(result_root)

    # 4.reselect
    _reselect_static_kernel()


def _compile_static_kernel_for_multi_card(fx_func, *args, gloo_group=None, is_cache_compile=False, build_dir=None,
                                          super_kernel_optimize=None, **kwargs):
    rank = dist.get_rank()
    result_root = safe_resolve_output_dir(build_dir)
    warnings.warn(
        f"Rank {rank}: starting static kernel compilation for multi card, the build directory is {result_root}")

    if not gloo_group:
        gloo_group = _get_local_gloo_group()

    # 1.执行单算子，用于生成算子信息json
    logger.debug(f"Rank {rank}: start to execute dump json")
    try:
        _fx_func_run(args, fx_func, kwargs, result_root, rank)
    except Exception as e:
        warnings.warn(f"Failed to execute dump json: {e}")
    chosen_dir = _get_dumpjson_dir_for_opcompile(result_root, is_cache_compile, rank)

    # 2.收集各张卡的待编译json目录
    logger.debug(f"Rank {rank}: gather json dir after execute dump json")
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = dist.get_rank(group=gloo_group)
    if local_rank == 0:
        gathered_json_dirs = [None] * local_world_size
    else:
        gathered_json_dirs = None
    dist.gather_object(obj=chosen_dir, object_gather_list=gathered_json_dirs, dst=_get_group_root_rank(rank), group=gloo_group)

    # 3.local rank 0进行静态编译并install
    if local_rank == 0:
        _merge_and_compile_install(gathered_json_dirs, result_root, rank, super_kernel_optimize=super_kernel_optimize)

    # 4.等待local rank 0执行完
    logger.debug(f"Rank {rank}: barrier after install (Gloo)")
    dist.barrier(group=gloo_group)

    # 5.执行reselect
    _reselect_static_kernel(rank)

    # 6.通信完成后销毁本地 gloo 分组，释放资源，防止重复创建
    logger.debug(f"Rank {rank}: barrier after reselect (Gloo)")
    dist.barrier(group=gloo_group)
    try:
        destroy_local_gloo_groups()
    except Exception as e:
        logger.warning(f"failed to destroy local gloo groups after install_and_sync_run_pkgs: {e}")


def _merge_and_compile_install(gathered_json_dirs: list[Path], result_root: Path, rank, super_kernel_optimize=None):
    logger.debug(f"Rank {rank}: start to execute static compile, json dirs:{gathered_json_dirs}")
    # 创建目录并合并json
    gathered_opcompile_dir = _merge_dump_json(gathered_json_dirs, result_root)
    if gathered_opcompile_dir is None:
        return

    # 静态编译
    compile_result = static_compile(result_root, gathered_opcompile_dir, rank, super_kernel_optimize=super_kernel_optimize)
    if not compile_result:
        return

    # 安装run包
    _install_run_packages(result_root, rank)


@debug_time(phase_name="[static kernel] merge jump json")
def _merge_dump_json(gathered_json_dirs: list[Path], result_root: Path) -> Union[Path, None]:
    gathered_opcompile_dir = result_root / f"{os.getpid()}_opcompile_gathered"
    try:
        gathered_opcompile_dir.mkdir(exist_ok=True)
    except Exception as e:
        warnings.warn(f"failed to create gathered opcompile directory {gathered_opcompile_dir}: {e}")
        return None

    _selected_json = set()
    _copied_json_filenames = set()
    filename_counter = {}
    for json_dir in gathered_json_dirs:
        if json_dir is None:
            continue
        json_files = list(json_dir.glob("*.json"))
        if not json_files:
            continue
        for json_file in json_files:
            json_hash = get_op_hash(json_file.resolve())
            if json_hash not in _selected_json:
                _selected_json.add(json_hash)
                try:
                    stem = json_file.stem
                    suffix = json_file.suffix
                    counter = filename_counter.get(stem, 0)
                    if counter == 0:
                        dest_filename = json_file.name
                    else:
                        dest_filename = f"{stem}_{counter}{suffix}"
                    while dest_filename in _copied_json_filenames:
                        counter += 1
                        dest_filename = f"{stem}_{counter}{suffix}"
                    _copied_json_filenames.add(dest_filename)
                    filename_counter[stem] = counter + 1
                    dest_path = gathered_opcompile_dir / dest_filename
                    shutil.copy2(json_file.resolve(), dest_path)
                except Exception as e:
                    warnings.warn(f"failed to copy {json_dir} to {gathered_opcompile_dir}: {e}")
                    return None
    g_json_files = list(gathered_opcompile_dir.glob("*.json"))
    if not g_json_files:
        logger.debug(f"No JSON files in {gathered_opcompile_dir}, skip op_compiler")
        return None
    else:
        return gathered_opcompile_dir


def _get_dumpjson_dir_for_opcompile(result_root: Path, compile_cache_dir=False, rank: int = None) -> Union[None, Path]:
    prefix = _get_log_prefix(rank)
    opcompile_dirs = [d for d in result_root.iterdir() if d.is_dir() and d.name.endswith("_opcompile")]
    if opcompile_dirs:
        if compile_cache_dir is False:
            compile_dir = get_compile_dir(opcompile_dirs[0])
            if not any(compile_dir.iterdir()):
                logger.debug(f"{prefix}Static compilation skipped: no operators in {compile_dir}")
                return None
            chosen_dir = compile_dir
            logger.debug(f"{prefix}Using compile directory: {chosen_dir}")
        else:
            # 对于cache compile场景，每次都需要全量编译
            chosen_dir = opcompile_dirs[0]
            logger.debug(f"{prefix}Using opcompile directory: {chosen_dir}")
    else:
        debug_dirs = [d for d in result_root.iterdir() if d.is_dir() and d.name.endswith("_debug")]
        if not debug_dirs:
            logger.debug("{prefix}Can not find json of ops, do not execute op_compiler")
            return None
        chosen_dir = debug_dirs[0]
        logger.debug(f"{prefix}Using debug directory: {chosen_dir}")
    json_files = list(chosen_dir.glob("*.json"))
    if not json_files:
        logger.debug(f"{prefix}No JSON files in {chosen_dir}, skip op_compiler")
        return None
    return chosen_dir


@debug_time(phase_name="[static kernel] static compile")
def static_compile(result_root: Path, chosen_dir, rank: int = None, super_kernel_optimize=None) -> bool:
    # 开始静态编译
    prefix = _get_log_prefix(rank)
    try:
        import torch_npu
        cmd = [
            "op_compiler",
            "-p", str(chosen_dir),
            "-v", torch_npu.npu.get_device_name(),
            "-j", _compute_opc_compile_jobs(),
            "-o", str(result_root),
        ]
        if super_kernel_optimize:
            cmd.extend(["--enable_super_kernel"])
        log_level = _get_compiler_log_level()
        if log_level is not None:
            cmd.extend(["-l", log_level])
        import torch_npu
        is_gte = torch_npu.npu.utils._is_gte_cann_version("9.0.0", module="CANN")
        if is_gte:
            cmd.extend(["-f", "true"])
        logger.debug(f"{prefix}the cmd is {cmd}")
        res = subprocess.run(cmd, check=True, capture_output=not is_gte, text=True)
        logger.debug(f"{prefix}execute op_compiler, msg: {res.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        warnings.warn(f"{prefix}execute op_compiler error, msg: {e.stderr}")
        return False


def _check_for_load_run_pkg(cached_cann_version, compile_cache_dir, cached_deterministic) -> bool:
    if cached_cann_version is None or compile_cache_dir is None:
        warnings.warn(
            "The cann_version and compile_cache_dir must be set, it is recommended to delete the compilation cache and recompile.")
        return False

    current_deterministic = "1" if torch.are_deterministic_algorithms_enabled() else "0"
    if cached_deterministic != current_deterministic:
        warnings.warn(
            f"The deterministic configuration has changed (from {cached_deterministic} to {current_deterministic}), "
            f"it is recommended to delete the compilation cache and recompile.")
        return False

    import torch_npu
    current_version = torch_npu.npu.utils.get_cann_version()
    if current_version != cached_cann_version:
        warnings.warn(
            f"There is a mismatch between the cached CANN version {cached_cann_version} and the current CANN version {current_version}, "
            f"it is recommended to delete the compilation cache and recompile.")
        return False

    if not os.path.exists(compile_cache_dir):
        warnings.warn(f"The path does not exist: {compile_cache_dir}")
        return False
    if os.stat(compile_cache_dir).st_uid != os.getuid():
        warnings.warn(f"Cache dir {compile_cache_dir} must be owned by the current user.")
        return False
    return True


def _find_and_install_run_pkgs(compile_cache_dir, rank: int = None) -> RunPackageStatus:
    prefix = _get_log_prefix(rank)
    compile_cache_dir_path = Path(compile_cache_dir)
    run_pkgs_dir = [d for d in compile_cache_dir_path.iterdir() if
                    d.is_dir() and d.name == "static_kernel_compile_outputs"]
    if len(run_pkgs_dir) == 0:
        return RunPackageStatus.NEED_COMPILE
    ts_outputs_dirs = [d for d in run_pkgs_dir[0].iterdir() if
                       d.is_dir() and d.name.endswith("_outputs") and d.name.startswith("ts")]
    if len(ts_outputs_dirs) == 0:
        return RunPackageStatus.NEED_COMPILE
    for ts_outputs_dir in ts_outputs_dirs:
        run_pkgs = list(ts_outputs_dir.glob("*.run"))
        if run_pkgs:
            logger.debug(f"{prefix}found {len(run_pkgs)} run files in {ts_outputs_dir.resolve()}.")
            _install_run_packages(ts_outputs_dir, rank)
            return RunPackageStatus.INSTALLED
    # has outputs dir, but does not have run, means static compile failed
    warnings.warn(
        f"{prefix}no run files were found, the static compilation might have failed, it is recommended to delete the compilation cache and recompile.")
    return RunPackageStatus.NEED_SKIP


@debug_time(phase_name="[static kernel] execute dump json")
def _fx_func_run(args, fx_func, kwargs, result_root, rank: int = None):
    import torch_npu
    torch_npu._C._aclop_start_dump(str(result_root))
    try:
        if isinstance(fx_func, torch.fx.GraphModule):
            torch.fx.Interpreter(fx_func).run(*args, **kwargs)
        else:
            fx_func(*args, **kwargs)
        import torch_npu
        torch_npu.npu.current_stream().synchronize()
    finally:
        torch_npu._C._aclop_stop_dump()
    prefix = _get_log_prefix(rank)
    logger.debug(f"{prefix}static kernel run eager success")


def _is_single_card() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True

    try:
        return dist.get_world_size() <= 1
    except Exception:
        return True


def _is_multicard_env_valid() -> bool:
    if not _is_single_card() and "LOCAL_WORLD_SIZE" not in os.environ:
        warnings.warn(
            "Environment variables 'LOCAL_WORLD_SIZE' is not set in a multi-card context. "
            "As a result, the static kernel feature will be disabled. "
            "To resolve this, please either launch the process via torchrun,"
            "or manually configure the 'LOCAL_WORLD_SIZE' environment variables. "
        )
        return False
    return True


@debug_time(phase_name="[static kernel] install static kernel run pkgs")
def _install_run_packages(result_root: Path, rank: int = None):
    run_pkgs = list(result_root.glob("*.run"))
    prefix = _get_log_prefix(rank)
    if not run_pkgs:
        logger.debug(f"{prefix}no static kernel run packages to install")
        return

    global _installed_run_pkgs
    for run_pkg in run_pkgs:
        run_pkg_hash = get_op_hash(run_pkg.resolve())
        if run_pkg_hash in _installed_run_pkgs:
            logger.info(f"Since {run_pkg.resolve()} is already installed, skipping installation.")
            continue
        filename = run_pkg.name
        try:
            res = subprocess.run([str(run_pkg)], check=True, capture_output=True, text=True)
            logger.debug(f"{prefix}{filename} install success, msg: {res.stdout}")
            save_uninstall_info(filename[:-4])
            _installed_run_pkgs.add(run_pkg_hash)
        except subprocess.CalledProcessError as e:
            logger.warning(f"{prefix}{filename} install failed, msg: {e.stderr}")


@debug_time(phase_name="[static kernel] reselect static kernel")
def _reselect_static_kernel(rank: int = None):
    prefix = _get_log_prefix(rank)
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
    global _local_gloo_groups
    _local_gloo_groups = [None] * node_count

    for node_idx in range(node_count):
        start = node_idx * local_world_size
        ranks = list(range(start, start + local_world_size))
        _local_gloo_groups[node_idx] = dist.new_group(ranks=ranks, backend="gloo")

    return _local_gloo_groups[global_rank // local_world_size]


def _get_group_root_rank(rank: int):
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    return (rank // local_world_size) * local_world_size


def destroy_local_gloo_groups():
    global _local_gloo_groups
    if not _local_gloo_groups:
        return

    for idx, pg in enumerate(_local_gloo_groups):
        if pg is None:
            continue
        try:
            dist.destroy_process_group(pg)
        except Exception as e:
            logger.warning(f"destroy_process_group failed for local group idx {idx}: {e}")
    _local_gloo_groups = None
    logger.debug("cleared _local_gloo_groups")


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

    base_output_dir = script_dir / "static_kernel_compile_outputs"
    try:
        base_output_dir.mkdir(exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"failed to create base output directory {base_output_dir}: {e}") from e

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    result_root = base_output_dir / f"ts{timestamp}_pid{os.getpid()}_outputs"

    try:
        result_root.mkdir(exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"failed to create output directory {result_root}: {e}") from e

    return result_root


def _compute_opc_compile_jobs():
    default = 16
    try:
        num_cpus = os.cpu_count() or 1
        jobs = num_cpus
        if jobs < 1 or jobs > num_cpus:
            return str(default)
        return str(jobs)
    except Exception as e:
        logger.warning("compute jobs failed, using default: %s", e)
        return str(default)


def get_compile_dir(opcompile_dir: Path) -> Path:
    """
    Get the directory used for static compilation.

    Args:
        opcompile_dir (Path): Directory of compilable operator JSON files.

    Returns:
        Path to the selected operator JSON files directory.
    """

    result_root = opcompile_dir.parent
    opcompile_selected_dir = result_root / f"{os.getpid()}_opcompile_selected"
    try:
        opcompile_selected_dir.mkdir(exist_ok=True)
    except Exception as e:
        logger.warning(f"Failed to create directory {opcompile_selected_dir}: {e}")
        logger.warning(f"Fallback to opcompile directory {opcompile_dir}")
        return opcompile_dir

    op_files = [f for f in opcompile_dir.iterdir() if f.is_file() and f.suffix == ".json"]
    for op_json in op_files:
        op_hash = get_op_hash(op_json)
        if op_hash not in _compiled_op_set:
            try:
                shutil.copy2(op_json, opcompile_selected_dir / op_json.name)
            except Exception as e:
                logger.warning(f"Failed to copy {op_json} to {opcompile_selected_dir}: {e}")
                logger.warning(f"Fallback to opcompile directory {opcompile_dir}")
                return opcompile_dir
            _compiled_op_set.add(op_hash)
        else:
            logger.debug(f"Operator described in {op_json.name} was already compiled, skipping static compilation.")
    return opcompile_selected_dir


def get_op_hash(op_json: Path, algo: str = "sha256") -> str:
    """
    Compute a content hash for an operator JSON file.

    Args:
        op_json (Path): Path to the operator JSON file.
        algo (str): Hash algorithm name supported by hashlib (default: sha256).

    Returns:
        Hexadecimal digest string.
    """
    hash_obj = hashlib.new(algo)
    with op_json.open("rb") as f:
        while chunk := f.read(65536):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def _get_compiler_log_level():
    log_level_map = {
        "0": "debug",
        "1": "info",
        "2": "warning",
        "3": "error",
        "4": None
    }

    env_level = os.getenv("ASCEND_GLOBAL_LOG_LEVEL", "3")
    return log_level_map.get(env_level, "error")


def _get_log_prefix(rank):
    prefix = f"Rank {rank}: " if rank is not None else ""
    return prefix
