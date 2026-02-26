import os

# Gen device sync before and after each fused kernel execution
_sync_around_fuse_kernel = os.getenv("ASCEND_LAUNCH_BLOCKING", None) == "1"

_debugging_host_only = False
