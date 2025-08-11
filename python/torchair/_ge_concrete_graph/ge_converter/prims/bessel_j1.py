from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.bessel_j1.default)
def conveter_prims_bessel_j1_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::bessel_j1(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.bessel_j1.default ge_converter is not implemented!")
