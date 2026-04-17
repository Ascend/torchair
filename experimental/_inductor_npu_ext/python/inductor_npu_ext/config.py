import os

# Gen device sync before and after each fused kernel execution
_sync_around_fuse_kernel = os.getenv("ASCEND_LAUNCH_BLOCKING", None) == "1"

_debug_options = set(os.getenv("TORCHINDUCTOR_NPU_EXT_DEBUG", "").split("+"))

_debugging_on_cpu = "cpu" in _debug_options or "nothrow" in _debug_options
