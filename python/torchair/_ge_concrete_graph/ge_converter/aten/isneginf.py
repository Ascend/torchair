from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.isneginf.default)
def conveter_aten_isneginf_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::isneginf(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.isneginf.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.isneginf.out)
def conveter_aten_isneginf_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::isneginf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.isneginf.out ge_converter is not implemented!")
