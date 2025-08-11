from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.nll_loss2d_backward.default)
def conveter_aten_nll_loss2d_backward_default(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: Union[int, Tensor],
    total_weight: Tensor,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.nll_loss2d_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.nll_loss2d_backward.grad_input)
def conveter_aten_nll_loss2d_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: Union[int, Tensor],
    total_weight: Tensor,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::nll_loss2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.nll_loss2d_backward.grad_input ge_converter is not implemented!")
