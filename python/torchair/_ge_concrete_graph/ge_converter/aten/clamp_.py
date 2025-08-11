from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.clamp_.default)
def conveter_aten_clamp__default(
    self: Tensor,
    min: Optional[Union[Number, Tensor]] = None,
    max: Optional[Union[Number, Tensor]] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp_.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_.Tensor)
def conveter_aten_clamp__Tensor(
    self: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::clamp_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp_.Tensor ge_converter is not implemented!")
