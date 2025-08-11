from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.bitwise_xor_.Tensor)
def conveter_aten_bitwise_xor__Tensor(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_xor_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bitwise_xor_.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_xor_.Scalar)
def conveter_aten_bitwise_xor__Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_xor_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bitwise_xor_.Scalar ge_converter is not implemented!")
