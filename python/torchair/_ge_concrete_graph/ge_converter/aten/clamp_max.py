from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.clamp_max.default)
def conveter_aten_clamp_max_default(
    self: Tensor, max: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_max(Tensor self, Scalar max) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.clamp_max.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_max.Tensor)
def conveter_aten_clamp_max_Tensor(self: Tensor, max: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::clamp_max.Tensor(Tensor self, Tensor max) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.clamp_max.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_max.out)
def conveter_aten_clamp_max_out(
    self: Tensor,
    max: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp_max.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_max.Tensor_out)
def conveter_aten_clamp_max_Tensor_out(
    self: Tensor, max: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_max.Tensor_out(Tensor self, Tensor max, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp_max.Tensor_out ge_converter is not implemented!")
