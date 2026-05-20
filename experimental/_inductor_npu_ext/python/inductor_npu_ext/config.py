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

_check_layout_enabled = os.getenv("TORCHINDUCTOR_NPU_EXT_LAYOUT_CHECK", "0") == "1"

_asc_cache_dir = os.getenv("TORCHINDUCTOR_NPU_EXT_CACHE_DIR", _default_asc_cache_dir())

# Enable TaskQueue instead of launching them synchronously from the calling thread
# Mode 0: synchronous mode (TASK_QUEUE_ENABLE=0)
# Mode 1: enqueue without clearing queue (TASK_QUEUE_ENABLE=1, recommended)
# Mode 2: enqueue after clearing queue (TASK_QUEUE_ENABLE=2)
_enable_taskqueue_mode = int(os.getenv("TASK_QUEUE_ENABLE", "1"))
if _enable_taskqueue_mode not in (0, 1, 2):
    raise ValueError(f"Invalid TASK_QUEUE_ENABLE={_enable_taskqueue_mode}, must be 0, 1, or 2")
