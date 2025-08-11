from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(2, 1152, 1, 1), F32(2, 1152, 1, 1)),
        Support(F32(96, 65), F32(96, 65)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.sigmoid_backward.default)
def conveter_aten_sigmoid_backward_default(
    grad_output: Tensor, output: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor"""
    return ge.SigmoidGrad(output, grad_output)


@register_fx_node_ge_converter(torch.ops.aten.sigmoid_backward.grad_input)
def conveter_aten_sigmoid_backward_grad_input(
    grad_output: Tensor,
    output: Tensor,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.sigmoid_backward.grad_input ge_converter is not implemented!")
