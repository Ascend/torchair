from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.lerp_.Scalar)
def conveter_aten_lerp__Scalar(
    self: Tensor, end: Tensor, weight: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.lerp_.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.lerp_.Tensor)
def conveter_aten_lerp__Tensor(
    self: Tensor, end: Tensor, weight: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.lerp_.Tensor ge_converter is not implemented!")
