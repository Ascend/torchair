from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.frac.default)
def conveter_aten_frac_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::frac(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.frac.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.frac.out)
def conveter_aten_frac_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.frac.out ge_converter is not implemented!")
