from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.rsqrt.default)
def conveter_aten_rsqrt_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::rsqrt(Tensor self) -> Tensor"""
    return ge.Rsqrt(self)


@register_fx_node_ge_converter(torch.ops.aten.rsqrt.out)
def conveter_aten_rsqrt_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.rsqrt.out ge_converter is not implemented!")
