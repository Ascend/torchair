from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.detach.default)
def conveter_aten_detach_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::detach(Tensor(a) self) -> Tensor(a)"""
    return ge.Identity(self)
