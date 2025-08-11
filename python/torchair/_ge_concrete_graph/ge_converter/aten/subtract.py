from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.subtract.Tensor)
def conveter_aten_subtract_Tensor(
    self: Tensor,
    other: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.subtract.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.subtract.out)
def conveter_aten_subtract_out(
    self: Tensor,
    other: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::subtract.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.subtract.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.subtract.Scalar)
def conveter_aten_subtract_Scalar(
    self: Tensor,
    other: Union[Number, Tensor],
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::subtract.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.subtract.Scalar ge_converter is not implemented!")
