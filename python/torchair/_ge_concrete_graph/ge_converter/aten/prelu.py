from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.prelu.default)
def conveter_aten_prelu_default(self: Tensor, weight: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::prelu(Tensor self, Tensor weight) -> Tensor"""
    return ge.PRelu(self, weight)
