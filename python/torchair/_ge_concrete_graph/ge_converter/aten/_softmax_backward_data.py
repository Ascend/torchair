from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(16), F32(16), -1, torch.float32),
    Support(F32(16, 16), F32(16, 16), 0, torch.float32),
])
@register_fx_node_ge_converter(torch.ops.aten._softmax_backward_data.default)
def conveter_aten__softmax_backward_data_default(
    grad_output: Tensor,
    output: Tensor,
    dim: int,
    input_dtype: int,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor"""
    input_ge_type = torch_type_to_ge_type(input_dtype)
    half_to_float = grad_output.dtype != input_ge_type
    if half_to_float:
        if grad_output.dtype != DataType.DT_FLOAT or \
                input_ge_type != DataType.DT_FLOAT16:
            raise RuntimeError("expected input and grad types to match,",
                               " or input to be at::Float and grad to be at::Half")
    return ge.SoftmaxGrad(output, grad_output, axes=[dim])


@register_fx_node_ge_converter(torch.ops.aten._softmax_backward_data.out)
def conveter_aten__softmax_backward_data_out(
    grad_output: Tensor,
    output: Tensor,
    dim: int,
    input_dtype: int,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise RuntimeError(
        "torch.ops.aten._softmax_backward_data.out is redundant before pytorch 2.1.0,"
        "might be supported in future version.")
