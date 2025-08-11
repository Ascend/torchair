from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.conj_physical.default)
def conveter_aten_conj_physical_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::conj_physical(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.conj_physical.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.conj_physical.out)
def conveter_aten_conj_physical_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::conj_physical.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.conj_physical.out ge_converter is not implemented!")
