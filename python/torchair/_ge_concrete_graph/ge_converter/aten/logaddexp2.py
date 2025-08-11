from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.logaddexp2.default)
def conveter_aten_logaddexp2_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::logaddexp2(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.logaddexp2.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.logaddexp2.out)
def conveter_aten_logaddexp2_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::logaddexp2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logaddexp2.out ge_converter is not implemented!")
