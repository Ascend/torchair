from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_type, \
torch_dtype_value_to_ge_proto_type, _ge_dtype_to_ge_proto_dtype

X_DTYPE_SUPPORT_LIST = {
    DataType.DT_FLOAT16,
    DataType.DT_BF16
}


@register_fx_node_ge_converter(torch.ops.npu.npu_all_to_all_matmul.default)
def convert_npu_all_to_all_matmul(
    x1: Tensor,
    x2: Tensor,
    hcom: str,
    world_size: int,
    bias: Optional[Tensor] = None,
    all2all_axes: Optional[List[int]] = None,
    all2all_out_flag: bool = True,
    meta_outputs: TensorSpec = None
):
    # transpose_x1 is set to False by default
    transpose_x1 = False
    transpose_x2 = False
    x1_scale = None
    x2_scale = None
    comm_scale = None
    x1_offset = None
    x2_offset = None
    x1_quant_mode = 0
    x2_quant_mode = 0
    comm_quant_mode = 0
    x1_quant_dtype = 28
    comm_quant_dtype = 28
    group_size = 0
    y_dtype_value = x1.dtype
    if all2all_axes is None:
        all2all_axes = [-2, -1]
    '''NB: npu::npu_all_to_all_matmul(Tensor x1, Tensor x2, str hcom, int world_size, Tensor? bias=None, int[]? all2all_axes=[-2, -1], bool all2all_out_flag=True) -> (Tensor, Tensor)'''
    check_dtype(x1, x2, bias)
    (out, all2all_out) = ge.AlltoAllMatmul(x1=x1,
                                           x2=x2,
                                           bias=bias,
                                           x1_scale=x1_scale,
                                           x2_scale=x2_scale,
                                           comm_scale=comm_scale,
                                           x1_offset=x1_offset,
                                           x2_offset=x2_offset,
                                           group=hcom,
                                           world_size=world_size,
                                           all2all_axes=all2all_axes,
                                           y_dtype=y_dtype_value,
                                           x1_quant_mode=x1_quant_mode,
                                           x2_quant_mode=x2_quant_mode,
                                           comm_quant_mode=comm_quant_mode,
                                           x1_quant_dtype=x1_quant_dtype,
                                           comm_quant_dtype=comm_quant_dtype,
                                           transpose_x1=transpose_x1,
                                           transpose_x2=transpose_x2,
                                           group_size=group_size,
                                           alltoall_out_flag=all2all_out_flag)
    out.desc.dtype = _ge_dtype_to_ge_proto_dtype(x1.dtype)
    all2all_out.desc.dtype = _ge_dtype_to_ge_proto_dtype(x1.dtype)
    return (out, all2all_out)


def check_dtype(x1: Tensor, x2: Tensor, bias: Tensor):
    if x1.dtype not in X_DTYPE_SUPPORT_LIST:
        raise AssertionError(f"Type of x1:{ge_type_to_torch_type(x1.dtype)} is error, x1 should be float16/bf16.")
    if x2.dtype not in X_DTYPE_SUPPORT_LIST:
        raise AssertionError(f"Type of x2:{ge_type_to_torch_type(x2.dtype)} is error, x2 should be float16/bf16.")
    if x1.dtype != x2.dtype:
        raise AssertionError(f"Type of x1:{ge_type_to_torch_type(x1.dtype)} and x2:{ge_type_to_torch_type(x2.dtype)} must be same.")
    if bias is not None:
        if bias.dtype != DataType.DT_FLOAT and bias.dtype != x1.dtype:
            raise AssertionError(f"Type of bias:{ge_type_to_torch_type(bias.dtype)} is error, bias should be {ge_type_to_torch_type(x1.dtype)} or float.")