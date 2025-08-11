from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.nextafter.default)
def conveter_aten_nextafter_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::nextafter(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.nextafter.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.nextafter.out)
def conveter_aten_nextafter_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::nextafter.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.nextafter.out ge_converter is not implemented!")
