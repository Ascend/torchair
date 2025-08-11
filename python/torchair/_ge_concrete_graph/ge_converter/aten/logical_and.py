from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 2), F32(2, 2)),
    Support(I32(2, 2), F32(2, 2)),
    Support(F64(2, 2), I64(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.logical_and.default)
def conveter_aten_logical_and_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::logical_and(Tensor self, Tensor other) -> Tensor"""
    self, other = dtype_promote(self, other, target_dtype=DataType.DT_BOOL)
    return ge.LogicalAnd(self, other)


@register_fx_node_ge_converter(torch.ops.aten.logical_and.out)
def conveter_aten_logical_and_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::logical_and.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logical_and.out ge_converter is not implemented!")
