from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.squeeze.default)
def conveter_prims_squeeze_default(
    a: Tensor, dimensions: List[int], meta_outputs: TensorSpec = None
):
    """NB: prims::squeeze(Tensor(a) a, int[] dimensions) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.prims.squeeze.default ge_converter is not implemented!")
