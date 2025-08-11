from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._reshape_alias.default)
def conveter_aten__reshape_alias_default(
    self: Tensor,
    size: Union[List[int], Tensor],
    stride: Union[List[int], Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_reshape_alias(Tensor(a) self, SymInt[] size, SymInt[] stride) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten._reshape_alias.default ge_converter is not implemented!")
