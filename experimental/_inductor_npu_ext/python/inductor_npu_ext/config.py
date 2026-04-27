import os
import getpass
import tempfile
import re


def _default_asc_cache_dir() -> str:
    sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
    return os.path.join(tempfile.gettempdir(), ".npu_kernels_" + sanitized_username)


# Gen device sync before and after each fused kernel execution
_sync_around_fuse_kernel = os.getenv("ASCEND_LAUNCH_BLOCKING", None) == "1"

_debug_options = set(os.getenv("TORCHINDUCTOR_NPU_EXT_DEBUG", "").split("+"))

_debugging_on_cpu = "cpu" in _debug_options or "nothrow" in _debug_options

_asc_cache_dir = os.getenv("TORCHINDUCTOR_NPU_EXT_CACHE_DIR", _default_asc_cache_dir())
