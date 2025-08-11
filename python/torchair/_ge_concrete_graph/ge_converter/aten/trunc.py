from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.trunc.default)
def conveter_aten_trunc_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::trunc(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.trunc.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.trunc.out)
def conveter_aten_trunc_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.trunc.out ge_converter is not implemented!")
