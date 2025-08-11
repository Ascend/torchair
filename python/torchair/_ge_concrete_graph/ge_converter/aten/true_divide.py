from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.true_divide.Tensor)
def conveter_aten_true_divide_Tensor(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.true_divide.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.true_divide.Scalar)
def conveter_aten_true_divide_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::true_divide.Scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.true_divide.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.true_divide.out)
def conveter_aten_true_divide_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::true_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.true_divide.out ge_converter is not implemented!")
