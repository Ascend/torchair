from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.arctan.default)
def conveter_aten_arctan_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::arctan(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.arctan.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.arctan.out)
def conveter_aten_arctan_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::arctan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.arctan.out ge_converter is not implemented!")
