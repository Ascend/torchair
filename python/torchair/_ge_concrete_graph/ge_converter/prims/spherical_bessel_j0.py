from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.spherical_bessel_j0.default)
def conveter_prims_spherical_bessel_j0_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::spherical_bessel_j0(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.spherical_bessel_j0.default ge_converter is not implemented!")
