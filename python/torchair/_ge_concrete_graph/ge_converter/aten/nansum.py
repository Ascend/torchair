from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.nansum.default)
def conveter_aten_nansum_default(
    self: Tensor,
    dim: Optional[List[int]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::nansum(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.nansum.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.nansum.out)
def conveter_aten_nansum_out(
    self: Tensor,
    dim: Optional[List[int]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::nansum.out(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.nansum.out ge_converter is not implemented!")
