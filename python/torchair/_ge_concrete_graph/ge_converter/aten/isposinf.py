from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.isposinf.default)
def conveter_aten_isposinf_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::isposinf(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.isposinf.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.isposinf.out)
def conveter_aten_isposinf_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::isposinf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.isposinf.out ge_converter is not implemented!")
