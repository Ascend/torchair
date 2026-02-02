from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, \
    torch_dtype_value_to_ge_type, torch_dtype_value_to_ge_proto_type, _ge_dtype_to_ge_proto_dtype


# x valid dtype list
X_DTYPE_SUPPORT_LIST = {
    DataType.DT_INT8,
    DataType.DT_HIFLOAT8,
    DataType.DT_FLOAT8_E5M2,
    DataType.DT_FLOAT8_E4M3FN
}

# scales valid dtype list
SCALES_DTYPE_SUPPORT_LIST = {
    DataType.DT_FLOAT32,
    DataType.DT_FLOAT8_E8M0
}


@declare_supported([
    Support(
        I8(128, 1024),
        F32(128, 1024),
        hcom="group",
        world_size=2,
        reduce_op="sum",
        output_dtype=None,
        x_dtype=None,
        scales_dtype=None  
    ),
])


@register_fx_node_ge_converter(torch.ops.npu.npu_quant_reduce_scatter.default)
def convert_npu_quant_reduce_scatter(
    x: Tensor,
    scales: Tensor,
    hcom: str,
    world_size: int,
    reduce_op: Optional[str] = "sum",
    output_dtype: Optional[int] = None,
    x_dtype: Optional[int] = None,
    scales_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """
    NB: npu_quant_reduce_scatter(Tensor x, Tensor scales, str hcom, int world_size, *,
                                 str? reduce_op='sum', int? output_dtype=None, int? x_dtype=None,
                                 int? scales_dtype=None) -> Tensor
    """
    if x_dtype is not None:
        x = ge.Bitcast(x, type=torch_dtype_value_to_ge_type(x_dtype))
        x.desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype)
    if scales_dtype is not None:
        scales = ge.Bitcast(scales, type=torch_dtype_value_to_ge_type(scales_dtype))
        scales.desc.dtype = torch_dtype_value_to_ge_proto_type(scales_dtype)
    if output_dtype is not None:
        output_dtype = torch_dtype_value_to_ge_type(output_dtype)
    check_dtype(x, scales)
    return ge.QuantReduceScatter(x=x,
                                 scales=scales,
                                 group=hcom,
                                 world_size=world_size,
                                 reduce_op=reduce_op,
                                 output_dtype=output_dtype)


def check_dtype(x: Tensor, scales: Tensor):
    if (x.dtype not in X_DTYPE_SUPPORT_LIST):
        raise AssertionError(f"The valid x dtype are: int8/hifloat8/float8_e5m2/float8_e4m3fn, but input value is: {x.dtype}")
    if (scales.dtype not in SCALES_DTYPE_SUPPORT_LIST):
        raise AssertionError(f"The valid scales dtype are: float32/hifloat8, but input value is: {scales.dtype}")
    if (x.dtype == DataType.DT_INT8 or x.dtype == DataType.DT_HIFLOAT8) and (scales.dtype == DataType.DT_FLOAT8_E8M0):
        raise AssertionError(f"When x dtype is int8/hifloat8, scales dtype cannot be float8_e8m0")
