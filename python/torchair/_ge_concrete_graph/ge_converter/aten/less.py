from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 3), F32(2, 3)),
    Support(F16(2, 3), F16(2, 3))
])
@register_fx_node_ge_converter(torch.ops.aten.less.Tensor)
def conveter_aten_less_Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::less.Tensor(Tensor self, Tensor other) -> Tensor"""
    if self.dtype != other.dtype:
        other = dtype_promote(other, target_dtype=self.dtype)
    return ge.Less(self, other)


@register_fx_node_ge_converter(torch.ops.aten.less.Scalar)
def conveter_aten_less_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::less.Scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.less.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.less.Scalar_out)
def conveter_aten_less_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::less.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.less.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.less.Tensor_out)
def conveter_aten_less_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::less.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.less.Tensor_out ge_converter is not implemented!")
