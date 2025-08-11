from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.softshrink.default)
def conveter_aten_softshrink_default(
    self: Tensor, lambd: Union[Number, Tensor] = 0.5, meta_outputs: TensorSpec = None
):
    """NB: aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.softshrink.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.softshrink.out)
def conveter_aten_softshrink_out(
    self: Tensor,
    lambd: Union[Number, Tensor] = 0.5,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.softshrink.out ge_converter is not implemented!")
