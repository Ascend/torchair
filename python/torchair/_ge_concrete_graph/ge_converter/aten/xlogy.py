from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.xlogy.Tensor)
def conveter_aten_xlogy_Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::xlogy.Tensor(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.xlogy.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.xlogy.Scalar_Other)
def conveter_aten_xlogy_Scalar_Other(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::xlogy.Scalar_Other(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.xlogy.Scalar_Other ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.xlogy.Scalar_Self)
def conveter_aten_xlogy_Scalar_Self(
    self: Union[Number, Tensor], other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::xlogy.Scalar_Self(Scalar self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.xlogy.Scalar_Self ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.xlogy.OutTensor)
def conveter_aten_xlogy_OutTensor(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::xlogy.OutTensor(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.xlogy.OutTensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.xlogy.OutScalar_Self)
def conveter_aten_xlogy_OutScalar_Self(
    self: Union[Number, Tensor],
    other: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::xlogy.OutScalar_Self(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.xlogy.OutScalar_Self ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.xlogy.OutScalar_Other)
def conveter_aten_xlogy_OutScalar_Other(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::xlogy.OutScalar_Other(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.xlogy.OutScalar_Other ge_converter is not implemented!")
