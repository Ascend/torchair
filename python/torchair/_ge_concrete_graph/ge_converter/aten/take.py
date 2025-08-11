from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.take.default)
def conveter_aten_take_default(self: Tensor, index: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::take(Tensor self, Tensor index) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.take.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.take.out)
def conveter_aten_take_out(
    self: Tensor, index: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.take.out ge_converter is not implemented!")
