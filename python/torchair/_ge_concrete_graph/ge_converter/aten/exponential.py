from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.exponential.default)
def conveter_aten_exponential_default(
    self: Tensor,
    lambd: float = 1.0,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::exponential(Tensor self, float lambd=1., *, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.exponential.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.exponential.out)
def conveter_aten_exponential_out(
    self: Tensor,
    lambd: float = 1.0,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::exponential.out(Tensor self, float lambd=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.exponential.out ge_converter is not implemented!")
