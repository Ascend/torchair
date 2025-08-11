from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.log10.default)
def conveter_prims_log10_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::log10(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.log10.default ge_converter is not implemented!")
