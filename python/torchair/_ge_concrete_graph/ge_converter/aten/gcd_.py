from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.gcd_.default)
def conveter_aten_gcd__default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::gcd_(Tensor(a!) self, Tensor other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.gcd_.default ge_converter is not implemented!")
