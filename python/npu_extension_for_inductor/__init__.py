import atexit
import functools
import os
import torch
import sys

from torch._inductor.lowering import lowerings
from torch._inductor.lowering import fallback_handler
from torch._inductor.codegen.common import register_backend_for_device
from torch._inductor.scheduler import Scheduler
from torch._ops import OpOverload, OpOverloadPacket
from npu_extension_for_inductor.npu import NPUScheduling, NpuWrapperCodeGen

register_backend_for_device("npu", NPUScheduling, NpuWrapperCodeGen)
if os.getenv("ASCIR_NOT_READY", None) == "1":
    register_backend_for_device("cpu", NPUScheduling, NpuWrapperCodeGen)


class LowerSummary:
    def __init__(self, fn):
        self.fn = fn
        self.lowered_ops = dict()
        self.fallback_ops = dict()

    def add_lowered(self, aten_op):
        self.lowered_ops[aten_op] = self.lowered_ops.get(aten_op, 0) + 1

    def add_fallback(self, aten_op):
        self.fallback_ops[aten_op] = self.fallback_ops.get(aten_op, 0) + 1

    def save(self):
        with open(self.fn, "w") as f:
            for aten_op, count in self.lowered_ops.items():
                f.write(f"{aten_op}: {count}\n")
                print(f"Lowered {aten_op}: {count}", flush=True)

            for aten_op, count in self.fallback_ops.items():
                f.write(f"Fallback {aten_op}: {count}\n")
                print(f"Fallback {aten_op}: {count}", flush=True)


LOWER_SUMMARY = LowerSummary("lower_summary.txt")
atexit.register(lambda: LOWER_SUMMARY.save())


def _wrap_npu(aten_fn, f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        LOWER_SUMMARY.add_lowered(aten_fn)
        return f(*args, **kwargs)

    return wrapper


def _wrap_fallback(aten_fn):
    def wrapper(*args, **kwargs):
        LOWER_SUMMARY.add_fallback(aten_fn)
        return fallback_handler(aten_fn, add_to_fallback_set=False)(*args, **kwargs)

    return wrapper


NPU_SUPPORTED_OPS_PREFIX = None

for op, lower_fn in lowerings.items():
    if NPU_SUPPORTED_OPS_PREFIX is not None:
        if isinstance(op, (OpOverload, OpOverloadPacket)) and \
                all(not str(op).startswith(prefix) for prefix in NPU_SUPPORTED_OPS_PREFIX):
            lowerings[op] = _wrap_fallback(op)
    else:
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
