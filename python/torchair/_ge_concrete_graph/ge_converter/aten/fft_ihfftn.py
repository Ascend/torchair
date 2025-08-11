from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.fft_ihfftn.default)
def conveter_aten_fft_ihfftn_default(
    self: Tensor,
    s: Optional[Union[List[int], Tensor]] = None,
    dim: Optional[List[int]] = None,
    norm: Optional[str] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::fft_ihfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.fft_ihfftn.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.fft_ihfftn.out)
def conveter_aten_fft_ihfftn_out(
    self: Tensor,
    s: Optional[Union[List[int], Tensor]] = None,
    dim: Optional[List[int]] = None,
    norm: Optional[str] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::fft_ihfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.fft_ihfftn.out ge_converter is not implemented!")
