from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import DataType, torch_dtype_value_to_ge_proto_type, torch_dtype_value_to_ge_type

def check_and_set_group_size(group_sizes):
    group_max = GROUP_SIZE_MAX_VALUE # group_m, group_n, group_k各占16位，组成64位group_siz, 因此每个值不能超过65535(16位的最大值)
    GROUP_M_OFFSET = 32
    GROUP_N_OFFSET = 16
    GROUP_SIZE_LEN = 3
    if(len(group_sizes) != GROUP_SIZE_LEN):
        raise RuntimeError("group_size must be a list with 3 elements, actual group_sizes is " + str(group_sizes))
    group_m = group_sizes[0]
    group_n = group_sizes[1]
    group_k = group_sizes[2]
    if (group_m > group_max or group_n > group_max or group_k > group_max):
        raise RuntimeError("group_size cannot be larger than 65535, actual group_sizes is " + str(group_sizes))
    if (group_m < 0 or group_n < 0 or group_k < 0):
        raise RuntimeError("group_size cannot be smaller than 0, actual group_sizes is " + str(group_sizes))
    group_size = (group_m << GROUP_M_OFFSET) + (group_n << GROUP_N_OFFSET) + group_k
    return group_size

@register_fx_node_ge_converter(torch.ops.npu.npu_transpose_quant_batchmatmul.default)
def conveter_npu_npu_transpose_quant_batchmatmul(
    x1: Tensor,
    x2: Tensor,
    dtype: int,
    bias: Optional[Tensor] = None,
    x1_scale: Optional[Tensor] = None,
    x2_scale: Optional[Tensor] = None,
    group_sizes: Optional[List[int]] = None,
    perm_x1: Optional[List[int]] = [1, 0, 2],
    perm_x2: Optional[List[int]] = [0, 1, 2],
    perm_y: Optional[List[int]] = [1, 0, 2],
    batch_split_factor: Optional[int] = 1,
    x1_dtype: Optional[int] = None,
    x2_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: npu::npu_transpose_quant_batchmatmul(Tensor x1, Tensor x2, int dtype, *, Tensor? bias=None,
    Tensor? x1_scale=None, Tensor? x2_scale=None, int[]? group_sizes=None, int[]? perm_x1=None,
    int[]? perm_x2=None, int[]? perm_y=None, int? batch_split_factor=1,
    int? x1_dtype=None, int? x2_dtype=None) -> Tensor
    """
    output_dtype = torch_dtype_value_to_ge_type(dtype)
    group_size = 0
    if group_sizes is not None and isinstance(group_sizes, List):
        group_size = check_and_set_group_size(group_sizes)
    if x1_dtype is not None:
        x1 = ge.Bitcast(x1, type=torch_dtype_value_to_ge_type(x1_dtype))
        x1.desc.dtype = torch_dtype_value_to_ge_proto_type(x1_dtype)
    if x2_dtype is not None:
        x2 = ge.Bitcast(x2, type=torch_dtype_value_to_ge_type(x2_dtype))
        x2.desc.dtype = torch_dtype_value_to_ge_proto_type(x2_dtype)
    out = ge.TransposeQuantBatchMatMul(
        x1, x2, bias=bias, x1_scale=x1_scale, x2_scale=x2_scale,
        dtype=output_dtype, group_size=group_size, perm_x1=perm_x1, perm_x2=perm_x2, perm_y=perm_y,
        batch_split_factor=batch_split_factor)
    out.desc.dtype = torch_dtype_value_to_ge_proto_type(dtype)
    return out
