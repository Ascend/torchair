from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(2, 8), 1.67, 1.05),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.elu.default)
def conveter_aten_elu_default(
    self: Tensor,
    alpha: Union[Number, Tensor] = 1,
    scale: Union[Number, Tensor] = 1,
    input_scale: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor"""
    return ge.Elu(self, alpha=alpha, scale=scale, input_scale=input_scale)


@register_fx_node_ge_converter(torch.ops.aten.elu.out)
def conveter_aten_elu_out(
    self: Tensor,
    alpha: Union[Number, Tensor] = 1,
    scale: Union[Number, Tensor] = 1,
    input_scale: Union[Number, Tensor] = 1,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.elu.out ge_converter is not supported!")
