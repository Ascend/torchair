from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.broadcast_in_dim.default)
def conveter_prims_broadcast_in_dim_default(
    a: Tensor,
    shape: Union[List[int], Tensor],
    broadcast_dimensions: List[int],
    meta_outputs: TensorSpec = None,
):
    """NB: prims::broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.prims.broadcast_in_dim.default ge_converter is not implemented!")
