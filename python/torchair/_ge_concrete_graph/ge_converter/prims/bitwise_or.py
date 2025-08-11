from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.bitwise_or.default)
def conveter_prims_bitwise_or_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: prims::bitwise_or(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.bitwise_or.default ge_converter is not implemented!")
