from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.subtract_.Tensor)
def conveter_aten_subtract__Tensor(
    self: Tensor,
    other: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::subtract_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.subtract_.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.subtract_.Scalar)
def conveter_aten_subtract__Scalar(
    self: Tensor,
    other: Union[Number, Tensor],
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::subtract_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.subtract_.Scalar ge_converter is not implemented!")
