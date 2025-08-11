from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.special_zeta.default)
def conveter_aten_special_zeta_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::special_zeta(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.special_zeta.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_zeta.other_scalar)
def conveter_aten_special_zeta_other_scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::special_zeta.other_scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.special_zeta.other_scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_zeta.self_scalar)
def conveter_aten_special_zeta_self_scalar(
    self: Union[Number, Tensor], other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::special_zeta.self_scalar(Scalar self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.special_zeta.self_scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_zeta.out)
def conveter_aten_special_zeta_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::special_zeta.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.special_zeta.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_zeta.self_scalar_out)
def conveter_aten_special_zeta_self_scalar_out(
    self: Union[Number, Tensor],
    other: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::special_zeta.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.special_zeta.self_scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_zeta.other_scalar_out)
def conveter_aten_special_zeta_other_scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::special_zeta.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.special_zeta.other_scalar_out ge_converter is not implemented!")
