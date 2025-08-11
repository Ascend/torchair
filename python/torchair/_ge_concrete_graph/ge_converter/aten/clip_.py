from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.clip_.default)
def conveter_aten_clip__default(
    self: Tensor,
    min: Optional[Union[Number, Tensor]] = None,
    max: Optional[Union[Number, Tensor]] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::clip_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clip_.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clip_.Tensor)
def conveter_aten_clip__Tensor(
    self: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::clip_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clip_.Tensor ge_converter is not implemented!")
