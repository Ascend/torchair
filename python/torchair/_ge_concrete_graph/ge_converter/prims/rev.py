from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.rev.default)
def conveter_prims_rev_default(a: Tensor, dims: List[int], meta_outputs: TensorSpec = None):
    """NB: prims::rev(Tensor a, int[] dims) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.rev.default ge_converter is not implemented!")
