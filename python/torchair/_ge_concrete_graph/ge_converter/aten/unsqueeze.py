from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(2, 2), dim=0),
        Support(F32(2, 2), dim=1),
        Support(F32(2, 2), dim=2),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.unsqueeze.default)
def conveter_aten_unsqueeze_default(self: Tensor, dim: int, meta_outputs: TensorSpec = None):
    """NB: aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)"""
    return ge.Unsqueeze(self, axes=[dim]);
