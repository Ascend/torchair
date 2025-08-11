from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.acos.default)
def conveter_prims_acos_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::acos(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.acos.default ge_converter is not implemented!")
