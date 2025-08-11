from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.logaddexp.default)
def conveter_aten_logaddexp_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::logaddexp(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.logaddexp.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.logaddexp.out)
def conveter_aten_logaddexp_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::logaddexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logaddexp.out ge_converter is not implemented!")
