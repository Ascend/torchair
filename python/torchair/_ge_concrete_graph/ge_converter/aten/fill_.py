from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.fill_.Scalar)
def conveter_aten_fill__Scalar(
    self: Tensor, value: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.fill_.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.fill_.Tensor)
def conveter_aten_fill__Tensor(self: Tensor, value: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.fill_.Tensor ge_converter is not implemented!")
