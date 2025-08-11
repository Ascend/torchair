from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.bitwise_right_shift.Tensor)
def conveter_aten_bitwise_right_shift_Tensor(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.bitwise_right_shift.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_right_shift.Tensor_Scalar)
def conveter_aten_bitwise_right_shift_Tensor_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.bitwise_right_shift.Tensor_Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_right_shift.Scalar_Tensor)
def conveter_aten_bitwise_right_shift_Scalar_Tensor(
    self: Union[Number, Tensor], other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.bitwise_right_shift.Scalar_Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_right_shift.Tensor_out)
def conveter_aten_bitwise_right_shift_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_right_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bitwise_right_shift.Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_right_shift.Tensor_Scalar_out)
def conveter_aten_bitwise_right_shift_Tensor_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_right_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bitwise_right_shift.Tensor_Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_right_shift.Scalar_Tensor_out)
def conveter_aten_bitwise_right_shift_Scalar_Tensor_out(
    self: Union[Number, Tensor],
    other: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_right_shift.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bitwise_right_shift.Scalar_Tensor_out ge_converter is not implemented!")
