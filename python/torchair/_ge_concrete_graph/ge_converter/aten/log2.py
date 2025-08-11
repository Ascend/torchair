from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(5)),
    Support(F16(5)),
    Support(I64(5)),
    Support(I32(5)),
    Support(I16(5)),
    Support(I8(5)),
])
@register_fx_node_ge_converter(torch.ops.aten.log2.default)
def conveter_aten_log2_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::log2(Tensor self) -> Tensor"""
    return ge.Log(self, base=2)


@register_fx_node_ge_converter(torch.ops.aten.log2.out)
def conveter_aten_log2_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.log2.out ge_converter is not implemented!")
