from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.abs.default)
def conveter_prims_abs_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::abs(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.abs.default ge_converter is not implemented!")
