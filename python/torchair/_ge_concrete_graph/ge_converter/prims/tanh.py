from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.tanh.default)
def conveter_prims_tanh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::tanh(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.tanh.default ge_converter is not implemented!")
