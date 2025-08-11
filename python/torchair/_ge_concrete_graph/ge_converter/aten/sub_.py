from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.sub_.Tensor)
def conveter_aten_sub__Tensor(
    self: Tensor,
    other: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.sub_.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sub_.Scalar)
def conveter_aten_sub__Scalar(
    self: Tensor,
    other: Union[Number, Tensor],
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.sub_.Scalar ge_converter is not implemented!")
