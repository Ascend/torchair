from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(4, 8)),
    Support(F32(4, 8)),
])
@register_fx_node_ge_converter(torch.ops.aten.gelu.default)
def conveter_aten_gelu_default(
    self: Tensor, *, approximate: str = "None", meta_outputs: TensorSpec = None
):
    """NB: aten::gelu(Tensor self, *, str approximate="none") -> Tensor"""
    return ge.Gelu(self)


@register_fx_node_ge_converter(torch.ops.aten.gelu.out)
def conveter_aten_gelu_out(
    self: Tensor,
    *,
    approximate: str = "None",
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::gelu.out(Tensor self, *, str approximate="none", Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError(
        "torch.ops.aten.gelu.out is redundant before pytorch 2.1.0,might be supported in future version.")
