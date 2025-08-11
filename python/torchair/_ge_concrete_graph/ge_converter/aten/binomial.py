from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.binomial.default)
def conveter_aten_binomial_default(
    count: Tensor,
    prob: Tensor,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.binomial.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.binomial.out)
def conveter_aten_binomial_out(
    count: Tensor,
    prob: Tensor,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::binomial.out(Tensor count, Tensor prob, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.binomial.out ge_converter is not implemented!")
