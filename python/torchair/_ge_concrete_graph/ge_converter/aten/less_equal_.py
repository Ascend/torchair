from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.less_equal_.Scalar)
def conveter_aten_less_equal__Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::less_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.less_equal_.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.less_equal_.Tensor)
def conveter_aten_less_equal__Tensor(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::less_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.less_equal_.Tensor ge_converter is not implemented!")
