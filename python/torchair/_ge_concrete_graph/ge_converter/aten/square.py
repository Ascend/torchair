from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.square.default)
def conveter_aten_square_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::square(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.square.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.square.out)
def conveter_aten_square_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::square.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.square.out ge_converter is not implemented!")
