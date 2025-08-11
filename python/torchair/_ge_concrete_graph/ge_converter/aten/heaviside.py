from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.heaviside.default)
def conveter_aten_heaviside_default(
    self: Tensor, values: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::heaviside(Tensor self, Tensor values) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.heaviside.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.heaviside.out)
def conveter_aten_heaviside_out(
    self: Tensor, values: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::heaviside.out(Tensor self, Tensor values, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.heaviside.out ge_converter is not implemented!")
