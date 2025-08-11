from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.absolute.default)
def conveter_aten_absolute_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::absolute(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.absolute.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.absolute.out)
def conveter_aten_absolute_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::absolute.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.absolute.out ge_converter is not implemented!")
