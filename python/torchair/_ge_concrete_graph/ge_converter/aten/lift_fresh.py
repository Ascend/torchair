from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.lift_fresh.default)
def conveter_aten_lift_fresh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::lift_fresh(Tensor(a) self) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.lift_fresh.default ge_converter is not implemented!")
