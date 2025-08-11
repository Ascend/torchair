from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.bitwise_and.default)
def conveter_prims_bitwise_and_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: prims::bitwise_and(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.bitwise_and.default ge_converter is not implemented!")
