from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(1, 16, 16)),
    Support(F32(1, 16, 16), diagonal=2),
    Support(F32(1, 16, 16), diagonal=-2),
    Support(F16(1, 16, 16)),
    Support(F16(1, 16, 16), diagonal=2),
    Support(F16(1, 16, 16), diagonal=-2),
    Support(BOOL(1, 16, 16)),
    Support(BOOL(1, 16, 16), diagonal=2),
    Support(BOOL(1, 16, 16), diagonal=-2),
])
@register_fx_node_ge_converter(torch.ops.aten.tril.default)
def conveter_aten_tril_default(
    self: Tensor, diagonal: int = 0, meta_outputs: TensorSpec = None
):
    """NB: aten::tril(Tensor self, int diagonal=0) -> Tensor"""
    return ge.Tril(self, diagonal=diagonal)


@register_fx_node_ge_converter(torch.ops.aten.tril.out)
def conveter_aten_tril_out(
    self: Tensor, diagonal: int = 0, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.tril.out ge_converter is not implemented!")
