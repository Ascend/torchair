from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.mish.default)
def conveter_aten_mish_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::mish(Tensor self) -> Tensor"""
    return ge.Mish(self)


@register_fx_node_ge_converter(torch.ops.aten.mish.out)
def conveter_aten_mish_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::mish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    out = ge.Mish(self)
    return out
