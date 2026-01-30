from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_fast_gelu_backward.default)
def converter_npu_fast_gelu_backward_default(
    grad: Tensor, 
    self: Tensor,
    meta_outputs: TensorSpec = None,
):
    """NB: func: npu_fast_gelu_backward(Tensor grad, Tensor self) -> Tensor
    """

    return ge.FastGeluGrad(grad, self)
