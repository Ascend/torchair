from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.float_power_.Tensor)
def conveter_aten_float_power__Tensor(
    self: Tensor, exponent: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::float_power_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.float_power_.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.float_power_.Scalar)
def conveter_aten_float_power__Scalar(
    self: Tensor, exponent: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::float_power_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.float_power_.Scalar ge_converter is not implemented!")
