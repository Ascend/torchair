from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.remainder.default)
def conveter_prims_remainder_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: prims::remainder(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.remainder.default ge_converter is not implemented!")
