from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.igammac_.default)
def conveter_aten_igammac__default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::igammac_(Tensor(a!) self, Tensor other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.igammac_.default ge_converter is not implemented!")
