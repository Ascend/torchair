from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.signbit.default)
def conveter_prims_signbit_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::signbit(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.signbit.default ge_converter is not implemented!")
