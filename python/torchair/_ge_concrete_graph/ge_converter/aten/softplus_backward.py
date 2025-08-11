from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.softplus_backward.default)
def conveter_aten_softplus_backward_default(
    grad_output: Tensor,
    self: Tensor,
    beta: Union[Number, Tensor],
    threshold: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.softplus_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.softplus_backward.grad_input)
def conveter_aten_softplus_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    beta: Union[Number, Tensor],
    threshold: Union[Number, Tensor],
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::softplus_backward.grad_input(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.softplus_backward.grad_input ge_converter is not implemented!")
