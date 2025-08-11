from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.replication_pad2d_backward.default)
def conveter_aten_replication_pad2d_backward_default(
    grad_output: Tensor,
    self: Tensor,
    padding: Union[List[int], Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::replication_pad2d_backward(Tensor grad_output, Tensor self, SymInt[4] padding) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.replication_pad2d_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.replication_pad2d_backward.grad_input)
def conveter_aten_replication_pad2d_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    padding: Union[List[int], Tensor],
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::replication_pad2d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.replication_pad2d_backward.grad_input ge_converter is not implemented!")
