from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support([F32(2, 2)]),
    Support([F16(2, 2), BF16(2, 2)]),
])
@register_fx_node_ge_converter(torch.ops.aten._foreach_sinh.default)
def conveter_aten__foreach_sinh_default(self: List[Tensor], meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_sinh(Tensor[] self) -> Tensor[]"""
    return ge.ForeachSinh(self)
