from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.fmin.default)
def conveter_aten_fmin_default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::fmin(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.fmin.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.fmin.out)
def conveter_aten_fmin_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::fmin.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.fmin.out ge_converter is not implemented!")
