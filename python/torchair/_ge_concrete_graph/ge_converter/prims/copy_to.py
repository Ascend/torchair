from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.copy_to.default)
def conveter_prims_copy_to_default(a: Tensor, b: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::copy_to(Tensor(a!) a, Tensor b) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.prims.copy_to.default ge_converter is not implemented!")
