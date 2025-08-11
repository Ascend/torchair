from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.fft_c2c.default)
def conveter_prims_fft_c2c_default(
    self: Tensor, *, dim: List[int], forward: bool, meta_outputs: TensorSpec = None
):
    """NB: prims::fft_c2c(Tensor self, *, int[] dim, bool forward) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.fft_c2c.default ge_converter is not implemented!")
