from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.special_i1e.default)
def conveter_aten_special_i1e_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::special_i1e(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.special_i1e.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_i1e.out)
def conveter_aten_special_i1e_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::special_i1e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.special_i1e.out ge_converter is not implemented!")
