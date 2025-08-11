from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(C64(2, 2)),
        Support(C64(2, 8, 16)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.view_as_real.default)
def conveter_aten_view_as_real_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::view_as_real(Tensor(a) self) -> Tensor(a)"""
    real = ge.Real(self)
    imag = ge.Imag(self)
    result = ge.Pack([real, imag], N=2, axis=-1)
    return result
