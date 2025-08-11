from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.mish.default)
def conveter_aten_mish_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::mish(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.mish.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.mish.out)
def conveter_aten_mish_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::mish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.mish.out ge_converter is not implemented!")
