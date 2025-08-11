from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.mvlgamma.default)
def conveter_aten_mvlgamma_default(self: Tensor, p: int, meta_outputs: TensorSpec = None):
    """NB: aten::mvlgamma(Tensor self, int p) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.mvlgamma.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.mvlgamma.out)
def conveter_aten_mvlgamma_out(
    self: Tensor, p: int, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::mvlgamma.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.mvlgamma.out ge_converter is not implemented!")
