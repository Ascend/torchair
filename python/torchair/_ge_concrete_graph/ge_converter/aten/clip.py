from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.clip.default)
def conveter_aten_clip_default(
    self: Tensor,
    min: Optional[Union[Number, Tensor]] = None,
    max: Optional[Union[Number, Tensor]] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.clip.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clip.Tensor)
def conveter_aten_clip_Tensor(
    self: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::clip.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.clip.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clip.out)
def conveter_aten_clip_out(
    self: Tensor,
    min: Optional[Union[Number, Tensor]] = None,
    max: Optional[Union[Number, Tensor]] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::clip.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clip.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clip.Tensor_out)
def conveter_aten_clip_Tensor_out(
    self: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::clip.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clip.Tensor_out ge_converter is not implemented!")
