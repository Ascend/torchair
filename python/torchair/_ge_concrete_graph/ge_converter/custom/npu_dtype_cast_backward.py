from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 2), dtype=torch.bool),
    Support(F16(16), dtype=torch.int32),
    Support(F32(8), dtype=torch.float16),
    Support(F16(4, 6), dtype=torch.float32),
    Support(F16(2, 1, 3, 4), dtype=torch.float16),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_dtype_cast_backward.default)
def converter_npu_dtype_cast_backward_default(
    self: Tensor,
    dtype: int,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_dtype_cast_backward(Tensor self, ScalarType dtype) -> Tensor"""
    return ge.Cast(self, dst_type=torch_type_to_ge_type(dtype))


@declare_supported([
    Support(F32(2, 2), dtype=torch.bool),
    Support(F16(16), dtype=torch.int32),
    Support(F32(8), dtype=torch.float16),
    Support(F16(4, 6), dtype=torch.float32),
    Support(F16(2, 1, 3, 4), dtype=torch.float16),
])
@register_fx_node_ge_converter(torch.ops.npu._npu_dtype_cast_backward.default)
def converter_old_npu_dtype_cast_backward_default(
    self: Tensor,
    dtype: int,
    meta_outputs: TensorSpec = None
):
    """NB: npu::_npu_dtype_cast_backward(Tensor self, ScalarType dtype) -> Tensor"""
    return ge.Cast(self, dst_type=torch_type_to_ge_type(dtype))
