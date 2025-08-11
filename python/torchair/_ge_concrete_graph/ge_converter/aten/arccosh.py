from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.arccosh.default)
def conveter_aten_arccosh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::arccosh(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.arccosh.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.arccosh.out)
def conveter_aten_arccosh_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::arccosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.arccosh.out ge_converter is not implemented!")
