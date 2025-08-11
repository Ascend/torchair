from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(3, 4), F32(3, 4)),
    Support(F16(3, 4), F16(3, 4)),
])
@register_fx_node_ge_converter(torch.ops.aten.hardsigmoid_backward.default)
def conveter_aten_hardsigmoid_backward_default(
    grad_output: Tensor, self: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::hardsigmoid_backward(Tensor grad_output, Tensor self) -> Tensor"""
    return ge.HardSigmoidGrad(grad_output, self)


@register_fx_node_ge_converter(torch.ops.aten.hardsigmoid_backward.grad_input)
def conveter_aten_hardsigmoid_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::hardsigmoid_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.hardsigmoid_backward.grad_input ge_converter is not supported!")
