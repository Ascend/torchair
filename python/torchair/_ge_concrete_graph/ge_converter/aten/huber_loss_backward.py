from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.huber_loss_backward.out)
def conveter_aten_huber_loss_backward_out(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    delta: float,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::huber_loss_backward.out(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.huber_loss_backward.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.huber_loss_backward.default)
def conveter_aten_huber_loss_backward_default(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    delta: float,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::huber_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.huber_loss_backward.default ge_converter is not implemented!")
