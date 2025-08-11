from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(3, 4), F32(3, 4), 0.4, False),
    Support(F16(3, 4), F16(3, 4), 0.4, False),
])
@register_fx_node_ge_converter(torch.ops.aten.leaky_relu_backward.default)
def conveter_aten_leaky_relu_backward_default(
    grad_output: Tensor,
    self: Tensor,
    negative_slope: Union[Number, Tensor],
    self_is_result: bool,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor"""
    if isinstance(negative_slope, Tensor):
        raise NotImplementedError('torch.ops.aten.leaky_relu_backward.default not supports negative_slope with Tensor type!')
    return ge.LeakyReluGrad(grad_output, self, negative_slope=negative_slope)


@register_fx_node_ge_converter(torch.ops.aten.leaky_relu_backward.grad_input)
def conveter_aten_leaky_relu_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    negative_slope: Union[Number, Tensor],
    self_is_result: bool,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::leaky_relu_backward.grad_input(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.leaky_relu_backward.grad_input ge_converter is not implemented!")
