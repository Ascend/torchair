from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.diag.default)
def conveter_aten_diag_default(
    self: Tensor, diagonal: int = 0, meta_outputs: TensorSpec = None
):
    """NB: aten::diag(Tensor self, int diagonal=0) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.diag.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.diag.out)
def conveter_aten_diag_out(
    self: Tensor, diagonal: int = 0, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::diag.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.diag.out ge_converter is not implemented!")
