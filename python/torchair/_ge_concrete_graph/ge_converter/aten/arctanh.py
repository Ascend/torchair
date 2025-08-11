from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.arctanh.default)
def conveter_aten_arctanh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::arctanh(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.arctanh.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.arctanh.out)
def conveter_aten_arctanh_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::arctanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.arctanh.out ge_converter is not implemented!")
