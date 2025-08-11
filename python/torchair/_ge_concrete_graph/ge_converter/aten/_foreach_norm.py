from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support([F32(2, 2, 2), F32(2, 3), F32(2, 3)], 1.),
        Support([F16(2, 2, 2), F16(2, 3), F16(2, 3)], 1.),
        Support([BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)], 1.),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_norm.Scalar)
def conveter_aten__foreach_norm_scalar(
    self: List[Tensor],
    scalar: Union[Number, Tensor] = 2,
    meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_norm.Scalar(Tensor[] self, Union[Number, Tensor] scalar) -> Tensor[]"""
    return ge.ForeachNorm(self, scalar)

