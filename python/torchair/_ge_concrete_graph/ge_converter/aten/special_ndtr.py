from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.special_ndtr.default)
def conveter_aten_special_ndtr_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::special_ndtr(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.special_ndtr.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_ndtr.out)
def conveter_aten_special_ndtr_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::special_ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.special_ndtr.out ge_converter is not implemented!")
