from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._foreach_mul_.Scalar)
def conveter_aten__foreach_mul__Scalar(
    self: List[Tensor], scalar: Union[Number, Tensor]
):
    """NB: aten::_foreach_mul_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_mul_.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_mul_.List)
def conveter_aten__foreach_mul__List(
    self: List[Tensor], other: List[Tensor]
):
    """NB: aten::_foreach_mul_.List(Tensor(a!)[] self, Tensor[] other) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_mul_.List ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_mul_.ScalarList)
def conveter_aten__foreach_mul__ScalarList(
    self: List[Tensor], scalars: Union[List[Number], Tensor]
):
    """NB: aten::_foreach_mul_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_mul_.ScalarList ge_converter is not implemented!")
