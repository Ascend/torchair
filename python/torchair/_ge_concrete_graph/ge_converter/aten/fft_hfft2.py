from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.fft_hfft2.default)
def conveter_aten_fft_hfft2_default(
    self: Tensor,
    s: Optional[Union[List[int], Tensor]] = None,
    dim: List[int] = (),
    norm: Optional[str] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::fft_hfft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.fft_hfft2.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.fft_hfft2.out)
def conveter_aten_fft_hfft2_out(
    self: Tensor,
    s: Optional[Union[List[int], Tensor]] = None,
    dim: List[int] = (),
    norm: Optional[str] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::fft_hfft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.fft_hfft2.out ge_converter is not implemented!")
