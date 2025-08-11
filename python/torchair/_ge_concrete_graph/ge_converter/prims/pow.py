from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.pow.default)
def conveter_prims_pow_default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::pow(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.pow.default ge_converter is not implemented!")
