from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.digamma.default)
def conveter_aten_digamma_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::digamma(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.digamma.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.digamma.out)
def conveter_aten_digamma_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.digamma.out ge_converter is not implemented!")
