from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.atan.default)
def conveter_prims_atan_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::atan(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.atan.default ge_converter is not implemented!")
