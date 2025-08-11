from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.dot.default)
def conveter_aten_dot_default(self: Tensor, tensor: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::dot(Tensor self, Tensor tensor) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.dot.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.dot.out)
def conveter_aten_dot_out(
    self: Tensor, tensor: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.dot.out ge_converter is not implemented!")
