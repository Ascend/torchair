from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.erfc.default)
def conveter_prims_erfc_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::erfc(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.erfc.default ge_converter is not implemented!")
