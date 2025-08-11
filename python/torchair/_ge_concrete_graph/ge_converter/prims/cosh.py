from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.cosh.default)
def conveter_prims_cosh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::cosh(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.cosh.default ge_converter is not implemented!")
