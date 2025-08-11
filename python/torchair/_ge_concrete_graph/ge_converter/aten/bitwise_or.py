from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(I32(2, 2), I32(2, 2)),
    Support(BOOL(2, 2), BOOL(2, 2)),
    Support(I32(2, 2), BOOL(2, 2)),
    Support(BOOL(2, 2), I32(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.bitwise_or.Tensor)
def conveter_aten_bitwise_or_Tensor(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor"""
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    if meta_outputs.dtype == DataType.DT_BOOL:
        output = ge.LogicalOr(self, other)
    else:
        output = ge.BitwiseOr(self, other)
    return output


@register_fx_node_ge_converter(torch.ops.aten.bitwise_or.Scalar_Tensor)
def conveter_aten_bitwise_or_Scalar_Tensor(
    self: Union[Number, Tensor], other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_or.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.bitwise_or.Scalar_Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_or.Scalar)
def conveter_aten_bitwise_or_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.bitwise_or.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_or.Tensor_out)
def conveter_aten_bitwise_or_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_or.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bitwise_or.Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_or.Scalar_out)
def conveter_aten_bitwise_or_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_or.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bitwise_or.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_or.Scalar_Tensor_out)
def conveter_aten_bitwise_or_Scalar_Tensor_out(
    self: Union[Number, Tensor],
    other: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_or.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bitwise_or.Scalar_Tensor_out ge_converter is not implemented!")
