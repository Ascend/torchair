from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(32, 768), F32(32, 768)),
    Support(F16(32, 768), F16(32, 768)),
])
@register_fx_node_ge_converter(torch.ops.aten.tanh_backward.default)
def conveter_aten_tanh_backward_default(
    grad_output: Tensor, output: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor"""
    return ge.TanhGrad(output, grad_output)


@register_fx_node_ge_converter(torch.ops.aten.tanh_backward.grad_input)
def conveter_aten_tanh_backward_grad_input(
    grad_output: Tensor,
    output: Tensor,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.tanh_backward.grad_input ge_converter is not supported!")
