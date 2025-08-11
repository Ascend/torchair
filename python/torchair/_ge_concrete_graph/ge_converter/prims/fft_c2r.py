from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.fft_c2r.default)
def conveter_prims_fft_c2r_default(
    self: Tensor,
    *,
    dim: List[int],
    last_dim_size: Union[int, Tensor],
    meta_outputs: TensorSpec = None
):
    """NB: prims::fft_c2r(Tensor self, *, int[] dim, SymInt last_dim_size) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.fft_c2r.default ge_converter is not implemented!")
