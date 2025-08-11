from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.exp2.default)
def conveter_prims_exp2_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::exp2(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.exp2.default ge_converter is not implemented!")
