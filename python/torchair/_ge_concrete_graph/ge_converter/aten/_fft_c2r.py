from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._fft_c2r.default)
def conveter_aten__fft_c2r_default(
    self: Tensor,
    dim: List[int],
    normalization: int,
    last_dim_size: Union[int, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_fft_c2r(Tensor self, int[] dim, int normalization, SymInt last_dim_size) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._fft_c2r.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._fft_c2r.out)
def conveter_aten__fft_c2r_out(
    self: Tensor,
    dim: List[int],
    normalization: int,
    last_dim_size: Union[int, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_fft_c2r.out(Tensor self, int[] dim, int normalization, SymInt last_dim_size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._fft_c2r.out ge_converter is not implemented!")
