from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.multiply.Tensor)
def conveter_aten_multiply_Tensor(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::multiply.Tensor(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.multiply.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.multiply.Scalar)
def conveter_aten_multiply_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::multiply.Scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.multiply.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.multiply.out)
def conveter_aten_multiply_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::multiply.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.multiply.out ge_converter is not implemented!")
