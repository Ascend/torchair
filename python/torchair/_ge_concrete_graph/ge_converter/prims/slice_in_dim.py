from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.slice_in_dim.default)
def conveter_prims_slice_in_dim_default(
    a: Tensor,
    start_index: Union[int, Tensor],
    limit_index: Union[int, Tensor],
    stride: int = 1,
    axis: int = 0,
    meta_outputs: TensorSpec = None,
):
    """NB: prims::slice_in_dim(Tensor(a) a, SymInt start_index, SymInt limit_index, int stride=1, int axis=0) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.prims.slice_in_dim.default ge_converter is not implemented!")
