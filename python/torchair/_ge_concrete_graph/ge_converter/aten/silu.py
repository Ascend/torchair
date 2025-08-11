from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.silu.default)
def conveter_aten_silu_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::silu(Tensor self) -> Tensor"""
    return ge.Swish(self)


@register_fx_node_ge_converter(torch.ops.aten.silu.out)
def conveter_aten_silu_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.silu.out ge_converter is not implemented!")
