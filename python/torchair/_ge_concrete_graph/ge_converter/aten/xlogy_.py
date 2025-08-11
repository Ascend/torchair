from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.xlogy_.Tensor)
def conveter_aten_xlogy__Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::xlogy_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.xlogy_.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.xlogy_.Scalar_Other)
def conveter_aten_xlogy__Scalar_Other(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::xlogy_.Scalar_Other(Tensor(a!) self, Scalar other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.xlogy_.Scalar_Other ge_converter is not implemented!")
