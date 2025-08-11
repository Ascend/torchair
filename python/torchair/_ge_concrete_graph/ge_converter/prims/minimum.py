from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.minimum.default)
def conveter_prims_minimum_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: prims::minimum(Tensor self, Tensor other) -> Tensor"""
    return ge.Minimum(self, other)
