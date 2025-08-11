from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support([F32(2, 2)]),
    Support([F16(2, 2), F16(3, 2)]),
])
@register_fx_node_ge_converter(torch.ops.aten._foreach_log1p.default)
def conveter_aten__foreach_log1p_default(self: List[Tensor], meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_log1p(Tensor[] self) -> Tensor[]"""
    return ge.ForeachLog1p(self)
