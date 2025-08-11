from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._prelu_kernel_backward.default)
def conveter_aten__prelu_kernel_backward_default(
    grad_output: Tensor, self: Tensor, weight: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::_prelu_kernel_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten._prelu_kernel_backward.default ge_converter is not implemented!")
