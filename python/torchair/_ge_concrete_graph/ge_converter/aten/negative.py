from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.negative.default)
def conveter_aten_negative_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::negative(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.negative.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.negative.out)
def conveter_aten_negative_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::negative.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.negative.out ge_converter is not implemented!")
