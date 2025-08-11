from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.rsqrt.default)
def conveter_prims_rsqrt_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::rsqrt(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.rsqrt.default ge_converter is not implemented!")
