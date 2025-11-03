from typing import List
import torch
from torch.fx.node import has_side_effect
from ._lib import lib


if not hasattr(torch.ops.air, "record"):
    lib.define("record(*, Device? device=None) -> Tensor")


    def record_impl(*, device=None) -> torch.Tensor:
        return torch.empty(0, device=device)


    torch.library.impl(lib, "record", "BackendSelect")(record_impl)


    @torch.library.impl(lib, "record", "Meta")
    def record_meta(*, device=None):
        return torch.empty(0, device='meta')


if not hasattr(torch.ops.air, "wait"):
    lib.define("wait(Tensor[] x) -> ()")
    has_side_effect(torch.ops.air.wait.default)


    def wait_impl(tensors):
        return None


    torch.library.impl(lib, "wait", "BackendSelect")(wait_impl)


    @torch.library.impl(lib, "wait", "Meta")
    def wait_meta(tensors):
        return None


def _wait(tensors: List[torch.Tensor]):
    if tensors is None or len(tensors) == 0:
        raise ValueError("torchair.ops.wait() requires at least one tensor input")
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("arguments of torchair.ops.wait() must be torch.Tensor")
    return torch.ops.air.wait(tensors)


def _record():
    return torch.ops.air.record()   