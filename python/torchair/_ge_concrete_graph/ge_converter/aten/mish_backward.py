from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.mish_backward.default)
def conveter_aten_mish_backward_default(
    grad_output: Tensor, self: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::mish_backward(Tensor grad_output, Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.mish_backward.default ge_converter is not implemented!")
