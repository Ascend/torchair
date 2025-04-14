import torch
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, ControlTensor, TensorSpec
from torchair._ge_concrete_graph import ge_apis as ge
from ._lib import lib

lib.define(
    """
    wait_tensor(Tensor self, Tensor dependency) -> Tensor
    """
)


@torch.library.impl(lib, "wait_tensor", "Meta")
def kernel_meta(self: torch.Tensor, dependency: torch.Tensor):
    return self


def kernel_impl(self: torch.Tensor, dependency: torch.Tensor):
    raise NotImplementedError("torch.ops.air.wait_tensor kernel_impl is not implemented!")


torch.library.impl(lib, "wait_tensor", "CPU")(kernel_impl)
torch.library.impl(lib, "wait_tensor", "PrivateUse1")(kernel_impl)


def _npu_wait_tensor(self: torch.Tensor, dependency: torch.Tensor):
    return torch.ops.air.wait_tensor(self, dependency)


@register_fx_node_ge_converter(torch.ops.air.wait_tensor.default)
def convert_wait_tensor(self: Tensor,
                        dependency: Tensor,
                        meta_outputs: TensorSpec = None
):
    control = ControlTensor(dependency.node).controller
    identity = ge.Identity(self)
    identity.node.input.append(control)
    return identity
