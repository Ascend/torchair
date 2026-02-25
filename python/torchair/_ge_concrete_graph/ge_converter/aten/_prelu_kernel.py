from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support([F32(2, 3, 4), F32(3)]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._prelu_kernel.default)
def conveter_aten__prelu_kernel_default(
    self: Tensor, weight: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::_prelu_kernel(Tensor self, Tensor weight) -> Tensor"""
    return ge.PRelu(self, weight)
