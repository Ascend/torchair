from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.lift.default)
def conveter_aten_lift_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::lift(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.lift.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.lift.out)
def conveter_aten_lift_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::lift.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.lift.out ge_converter is not implemented!")
