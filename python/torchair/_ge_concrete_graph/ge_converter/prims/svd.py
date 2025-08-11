from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.svd.default)
def conveter_prims_svd_default(
    A: Tensor, *, full_matrices: bool, meta_outputs: TensorSpec = None
):
    """NB: prims::svd(Tensor A, *, bool full_matrices) -> (Tensor U, Tensor S, Tensor Vh)"""
    raise NotImplementedError("torch.ops.prims.svd.default ge_converter is not implemented!")
