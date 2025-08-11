from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.multiply_.Tensor)
def conveter_aten_multiply__Tensor(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::multiply_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.multiply_.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.multiply_.Scalar)
def conveter_aten_multiply__Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::multiply_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.multiply_.Scalar ge_converter is not implemented!")
