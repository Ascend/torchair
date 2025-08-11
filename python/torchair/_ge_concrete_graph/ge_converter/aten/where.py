from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(BOOL(8, 4, 10, 10), F32(8, 4, 10, 10), F32(8, 4, 10, 10)),
        Support(BOOL(1, 1, 10, 10), F16(8, 4, 10, 10), F32(8, 4, 10, 10)),
        Support(BOOL(1, 1, 10, 10), I32(8, 4, 10, 10), F32(1, 1, 1, 10)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.where.self)
def conveter_aten_where_self(
    condition: Tensor, self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor"""
    if self.dtype != other.dtype:
        self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.SelectV2(condition, self, other)


@register_fx_node_ge_converter(torch.ops.aten.where.ScalarOther)
def conveter_aten_where_ScalarOther(
    condition: Tensor,
    self: Tensor,
    other: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor"""
    raise RuntimeError("torch.ops.aten.where.ScalarOther ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.where.ScalarSelf)
def conveter_aten_where_ScalarSelf(
    condition: Tensor,
    self: Union[Number, Tensor],
    other: Tensor,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor"""
    raise RuntimeError("torch.ops.aten.where.ScalarSelf ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.where.Scalar)
def conveter_aten_where_Scalar(
    condition: Tensor,
    self: Union[Number, Tensor],
    other: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::where.Scalar(Tensor condition, Scalar self, Scalar other) -> Tensor"""
    raise RuntimeError("torch.ops.aten.where.Scalar ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.where.default)
def conveter_aten_where_default(condition: Tensor, meta_outputs: List[TensorSpec] = None):
    """NB: aten::where(Tensor condition) -> Tensor[]"""
    raise RuntimeError("torch.ops.aten.where.default ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.where.self_out)
def conveter_aten_where_self_out(
    condition: Tensor,
    self: Tensor,
    other: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::where.self_out(Tensor condition, Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.where.self_out ge_converter is not supported!")
