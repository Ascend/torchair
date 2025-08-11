from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.deg2rad.default)
def conveter_aten_deg2rad_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::deg2rad(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.deg2rad.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.deg2rad.out)
def conveter_aten_deg2rad_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::deg2rad.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.deg2rad.out ge_converter is not implemented!")
