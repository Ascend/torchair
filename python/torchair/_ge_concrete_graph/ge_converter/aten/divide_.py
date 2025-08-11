from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.divide_.Tensor)
def conveter_aten_divide__Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.divide_.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.divide_.Tensor_mode)
def conveter_aten_divide__Tensor_mode(
    self: Tensor,
    other: Tensor,
    *,
    rounding_mode: Optional[str],
    meta_outputs: TensorSpec = None
):
    """NB: aten::divide_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.divide_.Tensor_mode ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.divide_.Scalar_mode)
def conveter_aten_divide__Scalar_mode(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    rounding_mode: Optional[str],
    meta_outputs: TensorSpec = None
):
    """NB: aten::divide_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.divide_.Scalar_mode ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.divide_.Scalar)
def conveter_aten_divide__Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.divide_.Scalar ge_converter is not implemented!")
