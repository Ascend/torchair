from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.log2.default)
def conveter_prims_log2_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::log2(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.log2.default ge_converter is not implemented!")
