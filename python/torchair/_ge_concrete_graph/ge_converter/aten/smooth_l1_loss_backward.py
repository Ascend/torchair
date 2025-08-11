from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.smooth_l1_loss_backward.grad_input)
def conveter_aten_smooth_l1_loss_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    beta: float,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::smooth_l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.smooth_l1_loss_backward.grad_input ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.smooth_l1_loss_backward.default)
def conveter_aten_smooth_l1_loss_backward_default(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    beta: float,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.smooth_l1_loss_backward.default ge_converter is not implemented!")
