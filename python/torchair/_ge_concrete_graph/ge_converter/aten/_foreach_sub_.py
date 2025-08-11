from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._foreach_sub_.Scalar)
def conveter_aten__foreach_sub__Scalar(
    self: List[Tensor], scalar: Union[Number, Tensor]
):
    """NB: aten::_foreach_sub_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_sub_.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_sub_.List)
def conveter_aten__foreach_sub__List(
    self: List[Tensor],
    other: List[Tensor],
    *,
    alpha: Union[Number, Tensor] = 1
):
    """NB: aten::_foreach_sub_.List(Tensor(a!)[] self, Tensor[] other, *, Scalar alpha=1) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_sub_.List ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_sub_.ScalarList)
def conveter_aten__foreach_sub__ScalarList(
    self: List[Tensor], scalars: Union[List[Number], Tensor]
):
    """NB: aten::_foreach_sub_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_sub_.ScalarList ge_converter is not implemented!")
