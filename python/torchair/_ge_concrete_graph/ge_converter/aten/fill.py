from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 2), 1),
])
@register_fx_node_ge_converter(torch.ops.aten.fill.Scalar)
def conveter_aten_fill_Scalar(
    self: Tensor, value: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::fill.Scalar(Tensor self, Scalar value) -> Tensor"""
    dims = ge.Shape(self)
    return ge.Fill(dims, ge.Cast(value, dst_type=self.dtype))


@register_fx_node_ge_converter(torch.ops.aten.fill.Scalar_out)
def conveter_aten_fill_Scalar_out(
    self: Tensor,
    value: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::fill.Scalar_out(Tensor self, Scalar value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.fill.Scalar_out ge_converter is redundant before pytorch 2.1.0!")


@declare_supported([
    Support(F32(2, 2), F32(1)),
    Support(F16(2, 2), F16(1)),
    Support(F16(4, 2), F16(1)),
])
@register_fx_node_ge_converter(torch.ops.aten.fill.Tensor)
def conveter_aten_fill_Tensor(self: Tensor, value: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::fill.Tensor(Tensor self, Tensor value) -> Tensor"""
    dims = ge.Shape(self)
    return ge.Fill(dims, ge.Cast(value, dst_type=self.dtype))


@register_fx_node_ge_converter(torch.ops.aten.fill.Tensor_out)
def conveter_aten_fill_Tensor_out(
    self: Tensor, value: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::fill.Tensor_out(Tensor self, Tensor value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.fill.Tensor_out ge_converter is redundant before pytorch 2.1.0!")
