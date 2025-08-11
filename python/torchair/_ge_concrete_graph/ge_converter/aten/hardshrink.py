from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.hardshrink.default)
def conveter_aten_hardshrink_default(
    self: Tensor, lambd: Union[Number, Tensor] = 0.5, meta_outputs: TensorSpec = None
):
    """NB: aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.hardshrink.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.hardshrink.out)
def conveter_aten_hardshrink_out(
    self: Tensor,
    lambd: Union[Number, Tensor] = 0.5,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::hardshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.hardshrink.out ge_converter is not implemented!")
