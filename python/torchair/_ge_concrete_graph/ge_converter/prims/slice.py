from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.slice.default)
def conveter_prims_slice_default(
    a: Tensor,
    start_indices: Union[List[int], Tensor],
    limit_indices: Union[List[int], Tensor],
    strides: Optional[Union[List[int], Tensor]] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: prims::slice(Tensor(a) a, SymInt[] start_indices, SymInt[] limit_indices, SymInt[]? strides=None) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.prims.slice.default ge_converter is not implemented!")
