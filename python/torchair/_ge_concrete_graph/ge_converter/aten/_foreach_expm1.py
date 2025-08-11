from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support([F32(2, 2, 2), F16(2, 3), BF16(2, 3)]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_expm1.default)
def conveter_aten__foreach_expm1_default(self: List[Tensor], meta_outputs: List[TensorSpec] = None):
    return ge.ForeachExpm1(self)
