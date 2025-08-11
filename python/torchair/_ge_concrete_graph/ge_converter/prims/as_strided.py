from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.as_strided.default)
def conveter_prims_as_strided_default(
    a: Tensor,
    size: Union[List[int], Tensor],
    stride: Union[List[int], Tensor],
    storage_offset: Union[int, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: prims::as_strided(Tensor(a!) a, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.prims.as_strided.default ge_converter is not implemented!")
