from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.masked_fill_.Scalar)
def conveter_aten_masked_fill__Scalar(
    self: Tensor, mask: Tensor, value: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)"""
    return ge.MaskedFill(self, mask, value)


@register_fx_node_ge_converter(torch.ops.aten.masked_fill_.Tensor)
def conveter_aten_masked_fill__Tensor(
    self: Tensor, mask: Tensor, value: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.masked_fill_.Tensor ge_converter is not implemented!")
