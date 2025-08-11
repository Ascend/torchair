from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.binary_cross_entropy.default)
def conveter_aten_binary_cross_entropy_default(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: int = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.binary_cross_entropy.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.binary_cross_entropy.out)
def conveter_aten_binary_cross_entropy_out(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: int = 1,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.binary_cross_entropy.out ge_converter is not implemented!")
