import os
import getpass
import tempfile
import re


def _default_asc_cache_dir() -> str:
    sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
    return os.path.join(tempfile.gettempdir(), ".npu_kernels_" + sanitized_username)


def _get_asc_cache_dir() -> str:
    cache_dir = os.getenv("TORCHINDUCTOR_NPU_EXT_CACHE_DIR")
    if cache_dir is None:
        return _default_asc_cache_dir()
    return os.path.abspath(os.path.expanduser(cache_dir))

asc_cache_dir = _get_asc_cache_dir()


# Gen device sync before and after each fused kernel execution
sync_around_fuse_kernel = os.getenv("ASCEND_LAUNCH_BLOCKING", None) == "1"

check_layout_enabled = os.getenv("TORCHINDUCTOR_NPU_EXT_LAYOUT_CHECK", "0") == "1"

debug_options = set(
    item for item in os.getenv("TORCHINDUCTOR_NPU_EXT_DEBUG", "").split("+")
    if item.strip()
)

_allowed_debug_options = {
    'cpu', # stub on cpu, mock asc codegen and runtime
    'decompose', # allow all decompose
    'lowering', # allow all native lowering
    'nocat', # disable cat lowering
    'nocanfuse', # disable canfuse
    'nothrow', # add an unsupported op rather than raise error on unsupported asc op
}

for opt in debug_options:
    if opt not in _allowed_debug_options:
        raise RuntimeError(f"Unknown debug option {opt}, allowed: {_allowed_debug_options}")

debugging_on_cpu = "cpu" in debug_options or "nothrow" in debug_options
disable_cat_fuse = "nocat" in debug_options
disable_canfuse = "nocanfuse" in debug_options

fuse_reduction_axis_threshold = 4096
