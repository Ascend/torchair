from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.transpose.default)
def conveter_prims_transpose_default(
    a: Tensor, permutation: List[int], meta_outputs: TensorSpec = None
):
    """NB: prims::transpose(Tensor(a) a, int[] permutation) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.prims.transpose.default ge_converter is not implemented!")
