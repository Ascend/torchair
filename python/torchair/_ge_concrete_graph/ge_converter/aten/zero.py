from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(8, 8)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.zero.default)
def conveter_aten_zero_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::zero(Tensor self) -> Tensor"""
    return ge.ZerosLike(self)


@register_fx_node_ge_converter(torch.ops.aten.zero.out)
def conveter_aten_zero_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::zero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.zero.out ge_converter is not implemented!")
