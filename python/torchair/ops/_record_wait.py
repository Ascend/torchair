from typing import List
import torch
from torch.fx.node import has_side_effect
from ._lib import lib


lib.define("record(*, Device? device=None) -> Tensor")


def record_impl(*, device=None) -> torch.Tensor:
    return torch.empty(0, device=device)


torch.library.impl(lib, "record", "BackendSelect")(record_impl)


@torch.library.impl(lib, "record", "Meta")
def record_meta(*, device=None):
    return torch.empty(0, device='meta')


lib.define("wait(Tensor[] x) -> ()")
has_side_effect(torch.ops.air.wait.default)


def wait_impl(tensors):
    return None


torch.library.impl(lib, "wait", "BackendSelect")(wait_impl)


@torch.library.impl(lib, "wait", "Meta")
def wait_meta(tensors):
    return None


def _wait(tensors: List[torch.Tensor]):
    return torch.ops.air.wait(tensors)


def _record():
    return torch.ops.air.record()   