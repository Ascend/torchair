from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.pow_.Scalar)
def conveter_aten_pow__Scalar(
    self: Tensor, exponent: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.pow_.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.pow_.Tensor)
def conveter_aten_pow__Tensor(self: Tensor, exponent: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.pow_.Tensor ge_converter is not implemented!")
