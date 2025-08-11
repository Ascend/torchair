from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul_.Scalar)
def conveter_aten__foreach_addcmul__Scalar(
    self: List[Tensor],
    tensor1: List[Tensor],
    tensor2: List[Tensor],
    value: Union[Number, Tensor] = 1
):
    """NB: aten::_foreach_addcmul_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_addcmul_.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul_.ScalarList)
def conveter_aten__foreach_addcmul__ScalarList(
    self: List[Tensor],
    tensor1: List[Tensor],
    tensor2: List[Tensor],
    scalars: Union[List[Number], Tensor]
):
    """NB: aten::_foreach_addcmul_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_addcmul_.ScalarList ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul_.Tensor)
def conveter_aten__foreach_addcmul__Tensor(
    self: List[Tensor],
    tensor1: List[Tensor],
    tensor2: List[Tensor],
    scalars: Tensor
):
    """NB: aten::_foreach_addcmul_.Tensor(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_addcmul_.Tensor ge_converter is not implemented!")
