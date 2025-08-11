from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.cumprod_.default)
def conveter_aten_cumprod__default(
    self: Tensor, dim: int, *, dtype: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::cumprod_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cumprod_.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cumprod_.dimname)
def conveter_aten_cumprod__dimname(
    self: Tensor, dim: str, *, dtype: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::cumprod_.dimname(Tensor(a!) self, str dim, *, ScalarType? dtype=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cumprod_.dimname ge_converter is not implemented!")
