from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.less_equal.Tensor)
def conveter_aten_less_equal_Tensor(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::less_equal.Tensor(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.less_equal.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.less_equal.Scalar)
def conveter_aten_less_equal_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::less_equal.Scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.less_equal.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.less_equal.Scalar_out)
def conveter_aten_less_equal_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::less_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.less_equal.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.less_equal.Tensor_out)
def conveter_aten_less_equal_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::less_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.less_equal.Tensor_out ge_converter is not implemented!")
