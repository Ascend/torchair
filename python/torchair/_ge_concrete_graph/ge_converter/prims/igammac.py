from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.igammac.default)
def conveter_prims_igammac_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: prims::igammac(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.igammac.default ge_converter is not implemented!")
