from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.expm1_.default)
def conveter_aten_expm1__default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::expm1_(Tensor(a!) self) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.expm1_.default ge_converter is not implemented!")
