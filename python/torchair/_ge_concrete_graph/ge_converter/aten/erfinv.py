from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.erfinv.default)
def conveter_aten_erfinv_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::erfinv(Tensor self) -> Tensor"""
    return ge.Erfinv(self)


@register_fx_node_ge_converter(torch.ops.aten.erfinv.out)
def conveter_aten_erfinv_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    out = ge.Erfinv(self)
    raise out
