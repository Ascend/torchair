from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.expm1.default)
def conveter_prims_expm1_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::expm1(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.expm1.default ge_converter is not implemented!")
