from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.arctan2_.default)
def conveter_aten_arctan2__default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::arctan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.arctan2_.default ge_converter is not implemented!")
