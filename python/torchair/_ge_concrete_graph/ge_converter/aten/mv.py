from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.mv.default)
def conveter_aten_mv_default(self: Tensor, vec: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::mv(Tensor self, Tensor vec) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.mv.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.mv.out)
def conveter_aten_mv_out(
    self: Tensor, vec: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::mv.out(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.mv.out ge_converter is not implemented!")
