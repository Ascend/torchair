from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.rrelu_with_noise_backward.default)
def conveter_aten_rrelu_with_noise_backward_default(
    grad_output: Tensor,
    self: Tensor,
    noise: Tensor,
    lower: Union[Number, Tensor],
    upper: Union[Number, Tensor],
    training: bool,
    self_is_result: bool,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.rrelu_with_noise_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rrelu_with_noise_backward.out)
def conveter_aten_rrelu_with_noise_backward_out(
    grad_output: Tensor,
    self: Tensor,
    noise: Tensor,
    lower: Union[Number, Tensor],
    upper: Union[Number, Tensor],
    training: bool,
    self_is_result: bool,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::rrelu_with_noise_backward.out(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.rrelu_with_noise_backward.out ge_converter is not implemented!")
