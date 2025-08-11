from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.special_spherical_bessel_j0.default)
def conveter_aten_special_spherical_bessel_j0_default(
    x: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::special_spherical_bessel_j0(Tensor x) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.special_spherical_bessel_j0.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_spherical_bessel_j0.out)
def conveter_aten_special_spherical_bessel_j0_out(
    x: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::special_spherical_bessel_j0.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.special_spherical_bessel_j0.out ge_converter is not implemented!")
