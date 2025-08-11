from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.signbit.default)
def conveter_aten_signbit_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::signbit(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.signbit.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.signbit.out)
def conveter_aten_signbit_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::signbit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.signbit.out ge_converter is not implemented!")
