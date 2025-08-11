from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.cauchy.default)
def conveter_aten_cauchy_default(
    self: Tensor,
    median: float = 0.0,
    sigma: float = 1.0,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::cauchy(Tensor self, float median=0., float sigma=1., *, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.cauchy.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cauchy.out)
def conveter_aten_cauchy_out(
    self: Tensor,
    median: float = 0.0,
    sigma: float = 1.0,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::cauchy.out(Tensor self, float median=0., float sigma=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cauchy.out ge_converter is not implemented!")
