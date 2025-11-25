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
    grad_dtype: int = None,
    input_dtype: int = None,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_dtype_cast_backward(Tensor self, ScalarType dtype, int? grad_dtype=None, int? input_dtype=None) -> Tensor"""
    if grad_dtype is not None:
        if grad_dtype == 296 or grad_dtype == 297:
            dim_num = self.rank
            bit_shape = []
            for _ in range(dim_num - 1):
                bit_shape.append(1)
            bit_shape.append(2)
            mul_num = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
            self_shape_uint8 = ge.Shape(self)
            self_shape_fp4 = ge.Mul(self_shape_uint8, mul_num)
            self = ge.Bitcast(self, type=torch_dtype_value_to_ge_type(grad_dtype))
            self = ge.Reshape(self, self_shape_fp4)
        else:
            self.desc.dtype = torch_dtype_value_to_ge_proto_type(grad_dtype)

    if input_dtype is not None:
        out = ge.Cast(self, dst_type=torch_dtype_value_to_ge_type(input_dtype))
        out.desc.dtype = torch_dtype_value_to_ge_proto_type(input_dtype)
        if input_dtype == 296 or input_dtype == 297:
            dim_num = self.rank
            bit_shape = []
            for _ in range(dim_num - 1):
                bit_shape.append(1)
            bit_shape.append(2)
            div_num = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
            out_shape_fp4 = ge.Shape(out)
            out_shape_uint8 = ge.Div(out_shape_fp4, div_num)
            out_shape_fp4_4bit = ge.ConcatV2([out_shape_uint8, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)], concat_dim=0, N=2)
            out = ge.Bitcast(ge.Reshape(out, out_shape_fp4_4bit), type=DataType.DT_UINT8)
            out = ge.Reshape(out, out_shape_uint8)
    else:
        out = ge.Cast(self, dst_type=torch_type_to_ge_type(dtype))

    return out


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
