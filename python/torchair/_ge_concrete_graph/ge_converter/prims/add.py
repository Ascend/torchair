from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.add.default)
def conveter_prims_add_default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::add(Tensor self, Tensor other) -> Tensor"""
    return ge.Add(self, other)
