from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.rad2deg.default)
def conveter_aten_rad2deg_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::rad2deg(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.rad2deg.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rad2deg.out)
def conveter_aten_rad2deg_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::rad2deg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.rad2deg.out ge_converter is not implemented!")
