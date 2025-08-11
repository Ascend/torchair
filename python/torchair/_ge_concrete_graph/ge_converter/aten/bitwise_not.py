from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(BOOL(2, 2)),
    Support(I32(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.bitwise_not.default)
def conveter_aten_bitwise_not_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::bitwise_not(Tensor self) -> Tensor"""
    if self.dtype == DataType.DT_BOOL:
        output = ge.LogicalNot(self)
    else:
        output = ge.Invert(self)
    return output


@register_fx_node_ge_converter(torch.ops.aten.bitwise_not.out)
def conveter_aten_bitwise_not_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.bitwise_not.out ge_converter is not supported!")
