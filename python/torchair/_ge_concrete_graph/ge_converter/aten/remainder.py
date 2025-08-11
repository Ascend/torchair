from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.remainder.Tensor)
def conveter_aten_remainder_Tensor(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor"""
    if self.dtype != other.dtype:
        self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.FloorMod(self, other)


@declare_supported([
    Support(F32(8, 12, 4096), 4096),
])
@register_fx_node_ge_converter(torch.ops.aten.remainder.Scalar)
def conveter_aten_remainder_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor"""
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.FloorMod(self, other)


@register_fx_node_ge_converter(torch.ops.aten.remainder.Scalar_Tensor)
def conveter_aten_remainder_Scalar_Tensor(
    self: Union[Number, Tensor], other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.FloorMod(self, other)


@register_fx_node_ge_converter(torch.ops.aten.remainder.Tensor_out)
def conveter_aten_remainder_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.remainder.Tensor_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.Scalar_out)
def conveter_aten_remainder_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.remainder.Scalar_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.Scalar_Tensor_out)
def conveter_aten_remainder_Scalar_Tensor_out(
    self: Union[Number, Tensor],
    other: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::remainder.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.remainder.Scalar_Tensor_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.int)
def conveter_aten_remainder_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::remainder.int(int a, int b) -> int"""
    raise RuntimeError("torch.ops.aten.remainder.int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.float)
def conveter_aten_remainder_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::remainder.float(float a, float b) -> float"""
    raise RuntimeError("torch.ops.aten.remainder.float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.int_float)
def conveter_aten_remainder_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::remainder.int_float(int a, float b) -> float"""
    raise RuntimeError("torch.ops.aten.remainder.int_float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.float_int)
def conveter_aten_remainder_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::remainder.float_int(float a, int b) -> float"""
    raise RuntimeError("torch.ops.aten.remainder.float_int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.default)
def conveter_aten_remainder_default(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::remainder(Scalar a, Scalar b) -> Scalar"""
    raise RuntimeError("torch.ops.aten.remainder.default ge_converter is not supported!")
