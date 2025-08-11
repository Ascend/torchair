from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(64, 4, 9), F32(64, 8, 9), 1),
        Support(F32(64, 4, 4), F32(64, 4, 8), 2),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.glu_backward.default)
def conveter_aten_glu_backward_default(
    grad_output: Tensor, self: Tensor, dim: int, meta_outputs: TensorSpec = None
):
    """NB: aten::glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor"""
    return ge.GLUGrad(grad_output, self, dim=dim)


@register_fx_node_ge_converter(torch.ops.aten.glu_backward.grad_input)
def conveter_aten_glu_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    dim: int,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::glu_backward.grad_input(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.glu_backward.grad_input ge_converter is not supported!")
