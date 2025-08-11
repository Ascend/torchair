from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.as_strided_.default)
def conveter_aten_as_strided__default(
    self: Tensor,
    size: Union[List[int], Tensor],
    stride: Union[List[int], Tensor],
    storage_offset: Optional[Union[int, Tensor]] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::as_strided_(Tensor(a!) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.as_strided_.default ge_converter is not implemented!")
