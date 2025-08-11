from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.arccos.default)
def conveter_aten_arccos_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::arccos(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.arccos.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.arccos.out)
def conveter_aten_arccos_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::arccos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.arccos.out ge_converter is not implemented!")
