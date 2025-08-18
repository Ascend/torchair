from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F16(16, 128), F16(32, 128), F16(32,)),

    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_linear.default)
def conveter_npu_npu_linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    """
    if input.dtype == DataType.DT_INT8 or weight.dtype == DataType.DT_INT8:
        raise RuntimeError("torch.ops.aten.npu_linear.default ge_converter is not support int8 dtype!")
    return ge.MatMul(input, weight, bias=bias, transpose_x1=False, transpose_x2=True)
