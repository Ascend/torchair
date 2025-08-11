from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.lt.default)
def conveter_prims_lt_default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::lt(Tensor self, Tensor other) -> Tensor"""
    return ge.Less(self, other)
