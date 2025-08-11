from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.abs.default)
def conveter_aten_abs_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::abs(Tensor self) -> Tensor"""
    if self.dtype == DataType.DT_UINT8:
        return self
    return ge.Abs(self)


@register_fx_node_ge_converter(torch.ops.aten.abs.out)
def conveter_aten_abs_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.abs.out ge_converter is not supported!")
