from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.atan2.default)
def conveter_prims_atan2_default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::atan2(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.atan2.default ge_converter is not implemented!")
