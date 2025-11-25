from torchair._ge_concrete_graph.ge_converter.converter_utils import *


def ge_type_to_torch_str(acl_type: int) -> str:
    if acl_type == DataType.DT_INT8 or acl_type == DataType.DT_QINT8:
        return "torch.qint8"
    if acl_type == DataType.DT_UINT8 or acl_type == DataType.DT_QUINT8:
        return "torch.quint8"
    if acl_type == DataType.DT_INT32 or acl_type == DataType.DT_QINT32:
        return "torch.qint32"
    if acl_type == DataType.DT_HIFLOAT8:
        return "torch.hifloat8"
    if acl_type == DataType.DT_FLOAT8_E5M2:
        return "torch.float8_e5m2"
    if acl_type == DataType.DT_FLOAT8_E4M3FN:
        return "torch.float8_e4m3fn"

    raise RuntimeError("Unsupported dst_type, only support torch.float8_e5m2, torch.float8_e4m3fn, torch_npu.hifloat8, "
                       "torch.int8, torch.qint8, torch.uint8, torch.quint8, torch.int32, torch.qint32 when div_mode=True")


@declare_supported([
    Support(F32(2, 2), F32(2), I32(2), dtype=torch.quint8, axis=1),
    Support(F16(2, 2), F32(2), I32(2), dtype=torch.qint32, axis=0),
    Support(F32(2, 2), F32(2), F32(2), dtype=torch.qint8, axis=-2, div_mode=False),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_quantize.default)
def conveter_npu_quantize_default(
    self: Tensor,
    scales: Tensor,
    zero_points: Optional[Tensor],
    dtype: int,
    axis: int = 1,
    div_mode: bool = True,
    meta_outputs: TensorSpec = None
):
    """
    NB: aten::npu_quantize(Tensor self, Tensor scales, 
                           Tensor? zero_points, int dtype, 
                           int axis=1, bool div_mode=True) -> Tensor
    """
    ge_dtype = torch_dtype_value_to_ge_type(dtype)

    if not div_mode:
        if axis > -1:
            axis = -1
        if ge_dtype == DataType.DT_QINT8:
            ge_dtype = DataType.DT_INT8
        y = ge.AscendQuantV2(self, scales, zero_points, sqrt_mode=False, round_mode="round",
                             dst_type=ge_dtype, axis=axis)
        y.desc.dtype = torch_dtype_value_to_ge_proto_type(dtype)
        if dtype == 16: # torch.quint4x2
            dim_num = self.rank
            bit_shape = []
            for _ in range(dim_num - 1):
                bit_shape.append(1)
            bit_shape.append(8)
            div_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
            y_shape_int4 = ge.Shape(y)
            y_shape_int32 = ge.Div(y_shape_int4, div_x2)
            y_shape_int4_4bit = ge.ConcatV2([y_shape_int32, ge.Cast(ge.Const([8]), dst_type=DataType.DT_INT32)],
                                            concat_dim=0, N=2)
            y = ge.Bitcast(ge.Reshape(y, y_shape_int4_4bit), type=DataType.DT_INT32)
            return ge.Reshape(y, y_shape_int32)
        else:
            return y

    if scales.rank != 1:
        raise RuntimeError("Scales' dim should be equal to 1.")
    if axis < 0:
        axis += self.rank

    dtype_str = ge_type_to_torch_str(ge_dtype)

    if axis < 0 or axis >= self.rank:
        raise RuntimeError("Axis should be in range [-rank(x), rank(x) - 1].")
    insert_dims = []
    for i in range(self.rank):
        if i != axis:
            insert_dims.append(i)
    if zero_points is not None:
        if zero_points.rank != 1:
            raise RuntimeError("Zero points' dim should be equal to 1.")
        zero_points = ge.Unsqueeze(zero_points, axes=insert_dims)
    scales = ge.Unsqueeze(scales, axes=insert_dims)
    y = ge.Quantize(self, scales=scales, zero_points=zero_points, dtype=dtype_str, axis=axis)
    y.desc.dtype = torch_dtype_value_to_ge_proto_type(dtype)
    return y