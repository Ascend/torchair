from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.nextafter.default)
def conveter_prims_nextafter_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: prims::nextafter(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.nextafter.default ge_converter is not implemented!")
