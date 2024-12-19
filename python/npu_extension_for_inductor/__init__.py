import functools
import os
import torch
import sys

from torch._inductor.lowering import lowerings
from torch._inductor.lowering import fallback_handler
from torch._inductor.codegen.common import register_backend_for_device
from torch._inductor.scheduler import Scheduler
from npu_extension_for_inductor.npu import NPUScheduling, NpuWrapperCodeGen
from npu_extension_for_inductor import lowering as npu_lowering

register_backend_for_device("cpu", NPUScheduling, NpuWrapperCodeGen)
register_backend_for_device("npu", NPUScheduling, NpuWrapperCodeGen)


def _wrap_npu(aten_fn, f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if os.getenv("NPU_INDUCTOR_ALWAYS_FALLBACK", None) == "1" and isinstance(aten_fn, torch._ops.OpOverload):
            return fallback_handler(aten_fn, add_to_fallback_set=False)(*args, **kwargs)

        return f(*args, **kwargs)

    return wrapper


for op, lower_fn in lowerings.items():
    lowerings[op] = _wrap_npu(op, lower_fn)


def patch_fn(model, fn):
    orig_fn = getattr(model, fn)

    def decorator(f):
        @functools.wraps(orig_fn)
        def inner(*args, **kwargs):
            return f(*args, **kwargs, orig_fn=orig_fn)

        setattr(model, fn, inner)
        return inner

    return decorator


@patch_fn(Scheduler, "can_fuse")
def can_fuse_vertical_npu(self, node1, node2, *, orig_fn):
    if NPUScheduling.can_fuse_npu(node1, node2):
        return True
    return orig_fn(self, node1, node2)
