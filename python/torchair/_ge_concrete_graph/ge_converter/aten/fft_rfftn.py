from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.fft_rfftn.default)
def conveter_aten_fft_rfftn_default(
    self: Tensor,
    s: Optional[Union[List[int], Tensor]] = None,
    dim: Optional[List[int]] = None,
    norm: Optional[str] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::fft_rfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.fft_rfftn.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.fft_rfftn.out)
def conveter_aten_fft_rfftn_out(
    self: Tensor,
    s: Optional[Union[List[int], Tensor]] = None,
    dim: Optional[List[int]] = None,
    norm: Optional[str] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::fft_rfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.fft_rfftn.out ge_converter is not implemented!")
