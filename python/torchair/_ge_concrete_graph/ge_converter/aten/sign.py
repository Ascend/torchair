from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(3, 3)),
        Support(F16(1,)),
        Support(I64(1,)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.sign.default)
def conveter_aten_sign_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sign(Tensor self) -> Tensor"""
    return ge.Sign(self)


@register_fx_node_ge_converter(torch.ops.aten.sign.out)
def conveter_aten_sign_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.sign.out ge_converter is not implemented!")
