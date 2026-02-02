from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_type, \
torch_dtype_value_to_ge_proto_type, _ge_dtype_to_ge_proto_dtype

X_DTYPE_SUPPORT_LIST = {
    DataType.DT_FLOAT8_E4M3FN,
    DataType.DT_FLOAT8_E5M2
}


@register_fx_node_ge_converter(torch.ops.npu.npu_quant_matmul_all_to_all.default)
def convert_npu_quant_matmul_all_to_all(
    x1: Tensor,
    x2: Tensor,
    hcom: str,
    world_size: int,
    bias: Optional[Tensor] = None,
    x1_scale: Optional[Tensor] = None,
    x2_scale: Optional[Tensor] = None,
    common_scale: Optional[Tensor] = None,
    x1_offset: Optional[Tensor] = None,
    x2_offset: Optional[Tensor] = None,
    x1_quant_mode: int = 3,
    x2_quant_mode: int = 2,
    common_quant_mode: int = 0,
    group_sizes: Optional[List[int]] = None,
    all2all_axes: Optional[List[int]] = None,
    comm_quant_dtype: int = 28,
    x1_dtype: int = None,
    x2_dtype: int = None,
    x1_scale_dtype: int = None,
    x2_scale_dtype: int = None,
    output_scale_dtype: int = None,
    comm_scale_dtype: int = None,
    y_dtype: int = None,
    meta_outputs: TensorSpec = None
):
    transpose_x1 = False
    transpose_x2 = False
    if y_dtype is None:
        raise RuntimeError("y_dtype is null")
    output_dtype = torch_dtype_value_to_ge_type(y_dtype)
    all2all_axes = [-1, -2]
    group_sizes = 0

    '''NB: npu::npu_quant_matmul_all_to_all(Tensor x1, Tensor x2, str hcom, int world_size, Tensor? bias=None, Tensor? x1_scale=None, Tensor? x2_scale=None,
    Tensor? common_scale=None, Tensor? x1_offset=None, Tensor? x1_offset=None, int? x1_quant_mode=0, int? x2_quant_mode=0, int? common_quant_mode=0,
    int[]? group_sizes=None, int[]? all2all_axes=[-1, -2]), int? comm_quant_dtype=28, int? x1_dtype=None, int? x2_dtype=None, int? x1_scale_dtype=None,
    int? x2_scale_dtype=None, int? output_scale_dtype=None, int? comm_scale_dtype=None, int? y_dtype=28) -> Tensor'''
    check_dtype(x1, x2, bias=bias, x1_scale=x1_scale, x2_scale=x2_scale)

    out = ge.MatmulAlltoAll(x1=x1,
                            x2=x2,
                            bias=bias,
                            x1_scale=x1_scale,
                            x2_scale=x2_scale,
                            comm_scale=common_scale,
                            x1_offset=x1_offset,
                            x2_offset=x2_offset,
                            group=hcom,
                            world_size=world_size,
                            all2all_axes=all2all_axes,
                            y_dtype=output_dtype,
                            x1_quant_mode=x1_quant_mode,
                            x2_quant_mode=x2_quant_mode,
                            comm_quant_mode=common_quant_mode,
                            comm_quant_dtype=comm_quant_dtype,
                            transpose_x1=transpose_x1,
                            transpose_x2=transpose_x2,
                            group_size=group_sizes)
    return out


def check_dtype(x1: Tensor, x2: Tensor, bias: Optional[Tensor], x1_scale: Optional[Tensor], x2_scale: Optional[Tensor]):
    if x1.dtype not in X_DTYPE_SUPPORT_LIST:
        raise AssertionError(f"Type of x1:{x1.dtype} is error, x1 should be [DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2].")
    if x2.dtype not in X_DTYPE_SUPPORT_LIST:
        raise AssertionError(f"Type of x2:{x2.dtype} is error, x2 should be [DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2].")
    if bias is not None:
        if bias.dtype != DataType.DT_FLOAT:
            raise AssertionError(f"if bias is not none, the data type of bias:{bias.dtype} should be DT_FLOAT.")
    if x1_scale is not None:
        if x1_scale.dtype != DataType.DT_FLOAT:
            raise AssertionError(f"if x1_scale is not none, the data type of x1_scale:{x1_scale.dtype} should be DT_FLOAT.")
    if x2_scale is not None:
        if x2_scale.dtype != DataType.DT_FLOAT:
            raise AssertionError(f"if x2_scale is not none, the data type of x2_scale:{x2_scale.dtype} should be DT_FLOAT.")