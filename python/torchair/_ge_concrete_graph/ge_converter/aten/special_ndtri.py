from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.special_ndtri.default)
def conveter_aten_special_ndtri_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::special_ndtri(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.special_ndtri.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_ndtri.out)
def conveter_aten_special_ndtri_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::special_ndtri.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.special_ndtri.out ge_converter is not implemented!")
