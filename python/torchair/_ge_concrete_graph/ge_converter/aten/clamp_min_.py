from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.clamp_min_.default)
def conveter_aten_clamp_min__default(
    self: Tensor, min: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp_min_.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_min_.Tensor)
def conveter_aten_clamp_min__Tensor(
    self: Tensor, min: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_min_.Tensor(Tensor(a!) self, Tensor min) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp_min_.Tensor ge_converter is not implemented!")
