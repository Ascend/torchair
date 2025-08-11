from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._standard_gamma.default)
def conveter_aten__standard_gamma_default(
    self: Tensor, generator: Optional[Generator] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::_standard_gamma(Tensor self, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._standard_gamma.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._standard_gamma.out)
def conveter_aten__standard_gamma_out(
    self: Tensor,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_standard_gamma.out(Tensor self, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._standard_gamma.out ge_converter is not implemented!")
