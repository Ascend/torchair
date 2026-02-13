from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_type, \
torch_dtype_value_to_ge_proto_type, _ge_dtype_to_ge_proto_dtype

X_DTYPE_SUPPORT_LIST = {
    DataType.DT_FLOAT8_E4M3FN,
    DataType.DT_FLOAT8_E5M2
}
Y_DTYPE_SUPPORT_LIST = {
    DataType.DT_FLOAT,
    DataType.DT_FLOAT16,
    DataType.DT_BF16
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
    if all2all_axes is None:
        all2all_axes = [-1, -2]

    if world_size not in [2, 4, 8, 16]:
        raise RuntimeError(f"world_size only supports value in [2, 4, 8, 16], but got {world_size}.")
    if x1_quant_mode != 3:
        raise RuntimeError(f"x1_quant_mode only supports value 3, whitch indicates dynamic per-token quantization, but got {x1_quant_mode}.")
    if x2_quant_mode != 2:
        raise RuntimeError(f"x2_quant_mode only supports value 2, whitch indicates per-channel quantization, but got {x2_quant_mode}.")
    if x1_scale is None:
        raise RuntimeError(f"per-token per-channel quantization need x1_scale, but got None.")
    if x2_scale is None:
        raise RuntimeError(f"per-token per-channel quantization need x2_scale, but got None.")
    
    if y_dtype is None:
        raise RuntimeError(f"per-token per-channel quantization need y_dtype, but got None.")
    output_dtype = torch_dtype_value_to_ge_type(y_dtype)
    if output_dtype not in Y_DTYPE_SUPPORT_LIST:
        raise RuntimeError(f"y_dtype should be {[ge_type_to_torch_type(d) for d in Y_DTYPE_SUPPORT_LIST]}, but got {y_dtype}.")

    '''NB: npu::npu_quant_matmul_all_to_all(Tensor x1, Tensor x2, str hcom, int world_size, Tensor? bias=None, Tensor? x1_scale=None, Tensor? x2_scale=None,
    Tensor? common_scale=None, Tensor? x1_offset=None, Tensor? x1_offset=None, int? x1_quant_mode=0, int? x2_quant_mode=0, int? common_quant_mode=0,
    int[]? group_sizes=None, int[]? all2all_axes=[-1, -2]), int? comm_quant_dtype=28, int? x1_dtype=None, int? x2_dtype=None, int? x1_scale_dtype=None,
    int? x2_scale_dtype=None, int? output_scale_dtype=None, int? comm_scale_dtype=None, int? y_dtype=28) -> Tensor'''
    check_dtype(x1, x2, bias=bias, x1_scale=x1_scale, x2_scale=x2_scale)

    group_max = 65535 # 65535是指group_size中的值最大不能超过16位可表示的范围
    group_size = 0
    if group_sizes is not None and isinstance(group_sizes, List):
        if(len(group_sizes) != 3):
            raise RuntimeError("group_size must be a list with 3 elements, actual group_sizes is " + str(group_sizes))
        group_m = group_sizes[0]
        group_n = group_sizes[1]
        group_k = group_sizes[2]
        if (group_m > group_max or group_n > group_max or group_k > group_max):
            raise RuntimeError("group_size can't large than 65535, actual group_sizes is " + str(group_sizes))
        if (group_m < 0 or group_n < 0 or group_k < 0):
            raise RuntimeError("group_size can't small than 0, actual group_sizes is " + str(group_sizes))
        group_size = (group_m << 32) + (group_n << 16) + group_k

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
                            group_size=group_size)
    return out


def check_dtype(x1: Tensor, x2: Tensor, bias: Optional[Tensor], x1_scale: Optional[Tensor], x2_scale: Optional[Tensor]):
    if x1.dtype not in X_DTYPE_SUPPORT_LIST:
        raise AssertionError(f"Type of x1:{ge_type_to_torch_type(x1.dtype)} is error, x1 should be {[ge_type_to_torch_type(d) for d in X_DTYPE_SUPPORT_LIST]}.")
    if x2.dtype not in X_DTYPE_SUPPORT_LIST:
        raise AssertionError(f"Type of x2:{ge_type_to_torch_type(x2.dtype)} is error, x2 should be {[ge_type_to_torch_type(d) for d in X_DTYPE_SUPPORT_LIST]}.")
    if bias is not None:
        if bias.dtype != DataType.DT_FLOAT:
            raise AssertionError(f"if bias is not none, the data type of bias:{ge_type_to_torch_type(bias.dtype)} should be float32.")
    if x1_scale is not None:
        if x1_scale.dtype != DataType.DT_FLOAT:
            raise AssertionError(f"if x1_scale is not none, the data type of x1_scale:{ge_type_to_torch_type(x1_scale.dtype)} should be float32.")
    if x2_scale is not None:
        if x2_scale.dtype != DataType.DT_FLOAT:
            raise AssertionError(f"if x2_scale is not none, the data type of x2_scale:{ge_type_to_torch_type(x2_scale.dtype)} should be float32.")