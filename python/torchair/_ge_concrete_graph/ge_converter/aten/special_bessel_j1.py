from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.special_bessel_j1.default)
def conveter_aten_special_bessel_j1_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::special_bessel_j1(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.special_bessel_j1.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_bessel_j1.out)
def conveter_aten_special_bessel_j1_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::special_bessel_j1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.special_bessel_j1.out ge_converter is not implemented!")
