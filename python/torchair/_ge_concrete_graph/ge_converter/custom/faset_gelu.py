from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_fast_gelu.default)
def conveter_npu_fast_gelu_default(
    x: Tensor,
    meta_outputs: TensorSpec = None,
):
    """NB: func: fast_gelu(Tensor x) -> Tensor"""
    """
    Warning: kernel [fast_gelu] is a out-of-place op, but it is supported by another in-place op cann.FastGelu.
    This current usage may cause the input to be changed unexpectedly, 
    and the caller needs to pay attention to this feature.
    """

    return ge.FastGelu(x)


@register_fx_node_ge_converter(torch.ops.npu.fast_gelu.default)
def conveter_fast_gelu_default(
    x: Tensor,
    meta_outputs: TensorSpec = None,
):
    """NB: func: fast_gelu(Tensor x) -> Tensor"""
    """
    Warning: kernel [fast_gelu] is a out-of-place op, but it is supported by another in-place op cann.FastGelu.
    This current usage may cause the input to be changed unexpectedly, 
    and the caller needs to pay attention to this feature.
    """

    return ge.FastGelu(x)
