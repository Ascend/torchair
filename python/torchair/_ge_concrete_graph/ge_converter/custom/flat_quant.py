from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(16, 64, 64), F16(64, 64), F16(64, 64)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_kronecker_quant.default)
def convert_npu_kronecker_quant(
    x: Tensor,
    kronecker_p1: Tensor,
    kronecker_p2: Tensor,
    clip_ratio: Optional[float] = 1.000000,
    dst_dtype: Optional[int] = torch.int32,
    meta_outputs: Any = None
):
    y_dtype = DataType.DT_INT32
    if dst_dtype is not None and dst_dtype != torch.int32:
        raise ValueError(f"dst_dtype should be int32, "
                         f"otherwise it should be None, but got {dst_dtype}")
    if clip_ratio is None:
        clip_ratio = 1.0
    y, quant_scale = ge.FlatQuant(x, kronecker_p1, kronecker_p2, clip_ratio=clip_ratio)
    dim_num = x.rank
    bit_shape = []
    for _ in range(dim_num - 1):
        bit_shape.append(1)
    bit_shape.append(8)
    # y int4 shape is (..., 8n), y int32 shape is (..., n), y bitcast shape is (..., n, 8)
    y_shape_int32 = ge.Div(ge.Shape(y), ge.Const(bit_shape, dtype=DataType.DT_INT32))
    y_shape_int4_bitcast = ge.ConcatV2([y_shape_int32, ge.Const([8], dtype=DataType.DT_INT32)], concat_dim=0, N=2)
    y = ge.Bitcast(ge.Reshape(y, y_shape_int4_bitcast), type=DataType.DT_INT32)
    return ge.Reshape(y, y_shape_int32), quant_scale