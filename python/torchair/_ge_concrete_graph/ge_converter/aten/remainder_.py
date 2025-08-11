from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.remainder_.Tensor)
def conveter_aten_remainder__Tensor(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.remainder_.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.remainder_.Scalar)
def conveter_aten_remainder__Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.remainder_.Scalar ge_converter is not implemented!")
