from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.lift_fresh_copy.default)
def conveter_aten_lift_fresh_copy_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::lift_fresh_copy(Tensor self) -> Tensor"""
    return ge.Identity(self)


@register_fx_node_ge_converter(torch.ops.aten.lift_fresh_copy.out)
def conveter_aten_lift_fresh_copy_out(
    self: Tensor, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::lift_fresh_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.lift_fresh_copy.out ge_converter is not implemented!")
