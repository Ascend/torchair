from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.fft_ifftshift.default)
def conveter_aten_fft_ifftshift_default(
    self: Tensor, dim: Optional[List[int]] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::fft_ifftshift(Tensor self, int[1]? dim=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.fft_ifftshift.default ge_converter is not implemented!")
