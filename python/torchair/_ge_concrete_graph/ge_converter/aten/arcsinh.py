from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.arcsinh.default)
def conveter_aten_arcsinh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::arcsinh(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.arcsinh.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.arcsinh.out)
def conveter_aten_arcsinh_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::arcsinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.arcsinh.out ge_converter is not implemented!")
