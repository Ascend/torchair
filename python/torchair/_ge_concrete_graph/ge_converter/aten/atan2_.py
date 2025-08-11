from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.atan2_.default)
def conveter_aten_atan2__default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.atan2_.default ge_converter is not implemented!")
