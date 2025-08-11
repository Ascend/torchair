from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.is_same_size.default)
def conveter_aten_is_same_size_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::is_same_size(Tensor self, Tensor other) -> bool"""
    raise NotImplementedError("torch.ops.aten.is_same_size.default ge_converter is not implemented!")
