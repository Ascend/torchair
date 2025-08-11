from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.conj_physical.default)
def conveter_prims_conj_physical_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::conj_physical(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.conj_physical.default ge_converter is not implemented!")
