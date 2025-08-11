from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(BOOL(2, 2), BOOL(2, 2)),
        Support(F32(2, 2), F32(2, 2)),
        Support(F16(2, 2), F16(2, 2)),
        Support(I32(2, 2), I16(2, 2)),
        Support(I32(2, 2), F16(2, 2))
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.logical_or.default)
def conveter_aten_logical_or_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::logical_or(Tensor self, Tensor other) -> Tensor"""
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.LogicalOr(self, other)


@register_fx_node_ge_converter(torch.ops.aten.logical_or.out)
def conveter_aten_logical_or_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::logical_or.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logical_or.out ge_converter is not implemented!")
