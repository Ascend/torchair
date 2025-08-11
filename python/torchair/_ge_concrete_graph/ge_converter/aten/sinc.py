from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.sinc.default)
def conveter_aten_sinc_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sinc(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.sinc.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sinc.out)
def conveter_aten_sinc_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.sinc.out ge_converter is not implemented!")
