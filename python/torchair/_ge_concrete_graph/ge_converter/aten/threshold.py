from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 2), 0., 0),
    Support(F32(2, 2), 0.1, 20),
    Support(F16(2, 2), 0., 0),
    Support(F16(2, 2), 0.1, 20),
])
@register_fx_node_ge_converter(torch.ops.aten.threshold.default)
def conveter_aten_threshold_default(
    self: Tensor,
    threshold: Union[Number, Tensor],
    value: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor"""
    return ge.ThresholdV2(self, threshold=threshold, value=value)


@register_fx_node_ge_converter(torch.ops.aten.threshold.out)
def conveter_aten_threshold_out(
    self: Tensor,
    threshold: Union[Number, Tensor],
    value: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.threshold.out ge_converter is not implemented!")
