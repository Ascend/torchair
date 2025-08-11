from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.cumprod.default)
def conveter_aten_cumprod_default(
    self: Tensor, dim: int, *, dtype: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.cumprod.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cumprod.dimname)
def conveter_aten_cumprod_dimname(
    self: Tensor, dim: str, *, dtype: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::cumprod.dimname(Tensor self, str dim, *, ScalarType? dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.cumprod.dimname ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cumprod.dimname_out)
def conveter_aten_cumprod_dimname_out(
    self: Tensor,
    dim: str,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::cumprod.dimname_out(Tensor self, str dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cumprod.dimname_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cumprod.out)
def conveter_aten_cumprod_out(
    self: Tensor,
    dim: int,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cumprod.out ge_converter is not implemented!")
