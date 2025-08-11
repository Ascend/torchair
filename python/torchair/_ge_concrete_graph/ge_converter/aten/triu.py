from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 2), 1)
])
@register_fx_node_ge_converter(torch.ops.aten.triu.default)
def conveter_aten_triu_default(
    self: Tensor, diagonal: int = 0, meta_outputs: TensorSpec = None
):
    """NB: aten::triu(Tensor self, int diagonal=0) -> Tensor"""
    return ge.Triu(self, diagonal=diagonal)


@register_fx_node_ge_converter(torch.ops.aten.triu.out)
def conveter_aten_triu_out(
    self: Tensor, diagonal: int = 0, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::triu.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.triu.out ge_converter is not supported!")
