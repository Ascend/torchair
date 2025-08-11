from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.round_.default)
def conveter_aten_round__default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::round_(Tensor(a!) self) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.round_.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.round_.decimals)
def conveter_aten_round__decimals(
    self: Tensor, *, decimals: int, meta_outputs: TensorSpec = None
):
    """NB: aten::round_.decimals(Tensor(a!) self, *, int decimals) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.round_.decimals ge_converter is not implemented!")
