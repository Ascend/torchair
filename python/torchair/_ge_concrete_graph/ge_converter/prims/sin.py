from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.sin.default)
def conveter_prims_sin_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::sin(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.sin.default ge_converter is not implemented!")
