from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.fmod.default)
def conveter_prims_fmod_default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::fmod(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.fmod.default ge_converter is not implemented!")
