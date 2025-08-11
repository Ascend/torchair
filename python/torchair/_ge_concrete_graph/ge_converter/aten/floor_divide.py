from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F64(7, 16, 16), F64(7, 16, 16, value_range=(1, 10))),
    Support(F32(7, 16, 16), F32(16, 16, value_range=(1, 10))),
    Support(F16(7, 16, 16), F16(16, 16, value_range=(1, 10))),
    Support(I64(7, 16, 16), I64(16, 16, value_range=(1, 10))),
    Support(I32(7, 16, 16), I32(16, 16, value_range=(1, 10))),
    Support(I16(7, 16, 16), I16(16, 16, value_range=(1, 10))),
    Support(I8(7, 16, 16), I8(16, 16, value_range=(1, 10))),
    Support(U8(7, 16, 16), U8(16, 16, value_range=(1, 10))),
    Support(F64(7, 16, 16), I64(16, 16, value_range=(1, 10))),
    Support(I32(7, 16, 16), U8(16, 16, value_range=(1, 10))),
])
@register_fx_node_ge_converter(torch.ops.aten.floor_divide.default)
def conveter_aten_floor_divide_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::floor_divide(Tensor self, Tensor other) -> Tensor"""
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.FloorDiv(self, other)


@register_fx_node_ge_converter(torch.ops.aten.floor_divide.Scalar)
def conveter_aten_floor_divide_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.floor_divide.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.floor_divide.out)
def conveter_aten_floor_divide_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.floor_divide.out ge_converter is not implemented!")
