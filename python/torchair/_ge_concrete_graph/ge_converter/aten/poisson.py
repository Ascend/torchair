from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.poisson.default)
def conveter_aten_poisson_default(
    self: Tensor, generator: Optional[Generator] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::poisson(Tensor self, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.poisson.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.poisson.out)
def conveter_aten_poisson_out(
    self: Tensor,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::poisson.out(Tensor self, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.poisson.out ge_converter is not implemented!")
