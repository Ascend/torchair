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
