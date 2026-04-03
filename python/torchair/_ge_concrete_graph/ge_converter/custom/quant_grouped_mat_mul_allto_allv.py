from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_type, \
torch_dtype_value_to_ge_proto_type, _ge_dtype_to_ge_proto_dtype

Y_DTYPE_SUPPORT_LIST = {
    DataType.DT_FLOAT16,
    DataType.DT_BF16
}

X_DTYPE_SUPPORT_LIST = {
    DataType.DT_HIFLOAT8
}

SCALE_DTYPE_SUPPORT_LIST = {
    DataType.DT_FLOAT
}

QUANT_MODE_SUPPORT_LIST = {
    1
}


@register_fx_node_ge_converter(torch.ops.npu.npu_quant_gmm_alltoallv.default)
def convert_npu_quant_gmm_alltoallv(
    gmm_x: Tensor,
    gmm_weight: Tensor,
    gmm_x_scale: Tensor,
    gmm_weight_scale: Tensor,
    hcom: str,
    ep_world_size: int,
    send_counts: List[int],
    recv_counts: List[int],
    gmm_y_dtype: int,
    *,
    send_counts_tensor: Optional[Tensor] = None,
    recv_counts_tensor: Optional[Tensor] = None,
    mm_x: Optional[Tensor] = None,
    mm_weight: Optional[Tensor] = None,
    mm_x_scale: Optional[Tensor] = None,
    mm_weight_scale: Optional[Tensor] = None,
    comm_quant_scale: Optional[Tensor] = None,
    gmm_x_quant_mode: int = None,
    gmm_weight_quant_mode: int = None,
    mm_x_quant_mode: int = None,
    mm_weight_quant_mode: int = None,
    comm_quant_mode: int = None,
    group_size: Optional[List[int]] = None,
    gmm_x_dtype: int = None,
    gmm_weight_dtype: int = None,
    gmm_x_scale_dtype: int = None,
    gmm_weight_scale_dtype: int = None,
    mm_x_dtype: int = None,
    mm_weight_dtype: int = None,
    mm_x_scale_dtype: int = None,
    mm_weight_scale_dtype: int = None,
    comm_quant_dtype: int = None,
    mm_y_dtype: int = None,
    meta_outputs: TensorSpec = None):
    '''
    npu_quant_gmm_alltoallv(Tensor gmm_x, Tensor gmm_weight, Tensor gmm_x_scale, Tensor gmm_weight_scale, 
    str hcom, int ep_world_size, int[] send_counts, int[] recv_counts, int gmm_y_dtype, *, 
    Tensor? send_counts_tensor=None, Tensor? recv_counts_tensor=None, Tensor? mm_x=None, 
    Tensor? mm_weight=None, Tensor? mm_x_scale=None, Tensor? mm_weight_scale=None,  
    Tensor? comm_quant_scale=None, int? gmm_x_quant_mode=None, int? gmm_weight_quant_mode=None, 
    int? mm_x_quant_mode=None, int? mm_weight_quant_mode=None, int? comm_quant_mode=None, 
    int[]? group_size=None, int? gmm_x_dtype=None, int? gmm_weight_dtype=None, 
    int? gmm_x_scale_dtype=None, int? gmm_weight_scale_dtype=None, int? mm_x_dtype=None, 
    int? mm_weight_dtype=None, int? mm_x_scale_dtype=None, int? mm_weight_scale_dtype=None, 
    int? comm_quant_dtype=None, int? mm_y_dtype=None) -> (Tensor, Tensor)
    '''
    dependencies = []
    node_name = None
    trans_gmm_weight = False
    trans_mm_weight = False
    group_sizes = 0
    comm_quant_dtype = 0
    gmm_x_quant_mode = 1 if gmm_x_quant_mode is None else gmm_x_quant_mode
    gmm_weight_quant_mode = 1 if gmm_weight_quant_mode is None else gmm_weight_quant_mode
    mm_x_quant_mode = 1 if mm_x_quant_mode is None else mm_x_quant_mode
    mm_weight_quant_mode = 1 if mm_weight_quant_mode is None else mm_weight_quant_mode

    tensor_names = ["gmm_x", "gmm_weight", "gmm_x_scale", "gmm_weight_scale", "mm_x", "mm_weight", \
                    "mm_x_scale", "mm_weight_scale"]
    dtype_names = ["gmm_x_dtype", "gmm_weight_dtype", "gmm_x_scale_dtype", "gmm_weight_scale_dtype", \
                   "mm_x_dtype", "mm_weight_dtype", "mm_x_scale_dtype", "mm_weight_scale_dtype"]
    quant_mode_names = ["gmm_x_quant_mode", "gmm_weight_quant_mode", "mm_x_quant_mode", \
                        "mm_weight_quant_mode"]

    x_list = [gmm_x, gmm_weight, gmm_x_scale, gmm_weight_scale, mm_x, mm_weight, mm_x_scale, mm_weight_scale]
    x_dtype_list = [gmm_x_dtype, gmm_weight_dtype, gmm_x_scale_dtype, gmm_weight_scale_dtype, mm_x_dtype, \
                    mm_weight_dtype, mm_x_scale_dtype, mm_weight_scale_dtype]
    quant_mode_list = [gmm_x_quant_mode, gmm_weight_quant_mode, mm_x_quant_mode, mm_weight_quant_mode]

    check_inputs(x_list, x_dtype_list, quant_mode_list, tensor_names, dtype_names, quant_mode_names)
    
    x_list = cast_to_real_type(x_list, x_dtype_list)
    (gmm_x, gmm_weight, gmm_x_scale, gmm_weight_scale, mm_x, mm_weight, mm_x_scale, mm_weight_scale, comm_quant_scale) = x_list
    output_dtype = torch_dtype_value_to_ge_type(gmm_y_dtype)
    if output_dtype not in Y_DTYPE_SUPPORT_LIST:
        raise RuntimeError( \
            f"gmm_y_dtype should be {[d for d in Y_DTYPE_SUPPORT_LIST]}, but got {output_dtype}")
    mm_out_dtype = None if mm_y_dtype is None else torch_dtype_value_to_ge_type(mm_y_dtype)
    if mm_out_dtype is not None and mm_out_dtype not in Y_DTYPE_SUPPORT_LIST:
        raise RuntimeError( \
            f"mm_y_dtype should be {[d for d in Y_DTYPE_SUPPORT_LIST]}, but got {mm_out_dtype}")
    if mm_out_dtype is None:
        mm_out_dtype = DataType.DT_FLOAT16

    (y, mm_y) = ge.QuantGroupedMatMulAlltoAllv(
        gmm_x=gmm_x, 
        gmm_weight=gmm_weight, 
        send_counts_tensor=send_counts_tensor, 
        recv_counts_tensor=recv_counts_tensor, 
        mm_x=mm_x, 
        mm_weight=mm_weight, 
        gmm_x_scale=gmm_x_scale, 
        gmm_weight_scale=gmm_weight_scale, 
        mm_x_scale=mm_x_scale, 
        mm_weight_scale=mm_weight_scale, 
        comm_quant_scale=comm_quant_scale, 
        group=hcom, 
        ep_world_size=ep_world_size, 
        send_counts=send_counts, 
        recv_counts=recv_counts, 
        trans_gmm_weight=trans_gmm_weight, 
        trans_mm_weight=trans_mm_weight, 
        gmm_x_quant_mode=gmm_x_quant_mode, 
        gmm_weight_quant_mode=gmm_weight_quant_mode, 
        mm_x_quant_mode=mm_x_quant_mode, 
        mm_weight_quant_mode=mm_weight_quant_mode, 
        comm_quant_mode=comm_quant_mode, 
        group_size=group_sizes, 
        comm_quant_dtype=comm_quant_dtype, 
        y_dtype=output_dtype, 
        mm_dtype=mm_out_dtype, 
        dependencies=dependencies, 
        node_name=node_name)

    y.desc.dtype = _ge_dtype_to_ge_proto_dtype(output_dtype)
    if mm_x is not None and mm_y is not None:
        mm_y.desc.dtype = _ge_dtype_to_ge_proto_dtype(mm_out_dtype)

    return (y, mm_y)


def cast_to_real_type(x_list, x_dtype_list):
    for i in range(len(x_list)):
        if x_dtype_list[i] is not None and x_list[i] is not None:
            x_list[i] = ge.Bitcast(x_list[i], type=torch_dtype_value_to_ge_type(x_dtype_list[i]))
            x_list[i].desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype_list[i])
    return x_list


def check_inputs(tensor_list, dtype_list, quant_mode_list, tensor_names, dtype_names, quant_mode_names):
    for input_tensor, input_dtype, tensor_name, dtype_name in \
        zip(tensor_list, dtype_list, tensor_names, dtype_names):
        if "gmm" in tensor_name:
            if input_tensor is None:
                raise RuntimeError( \
                    f"{tensor_name} should not be None")
        if input_tensor is not None and input_tensor.rank == 0:
            raise RuntimeError( \
                f"{tensor_name} dim num should not be 0")
        if "scale" not in tensor_name:
            if input_dtype is not None:
                ge_dtype = torch_dtype_value_to_ge_type(input_dtype)
                if ge_dtype not in X_DTYPE_SUPPORT_LIST:
                    raise RuntimeError( \
                        f"{dtype_name} should be {[d for d in X_DTYPE_SUPPORT_LIST]}, but got {ge_dtype}")
            else:
                if input_tensor is not None and input_tensor.dtype not in X_DTYPE_SUPPORT_LIST:
                    raise RuntimeError( \
                        f"{tensor_name}'s dtype should be {[d for d in X_DTYPE_SUPPORT_LIST]}, " + \
                            f"but got {input_tensor.dtype}")
        else:
            if input_dtype is not None:
                ge_dtype = torch_dtype_value_to_ge_type(input_dtype)
                if ge_dtype not in SCALE_DTYPE_SUPPORT_LIST:
                    raise RuntimeError( \
                        f"{dtype_name} should be {[d for d in SCALE_DTYPE_SUPPORT_LIST]}, but got {ge_dtype}")
            else:
                if input_tensor is not None and input_tensor.dtype not in SCALE_DTYPE_SUPPORT_LIST:
                    raise RuntimeError( \
                        f"{tensor_name}'s dtype should be {[d for d in SCALE_DTYPE_SUPPORT_LIST]}, " + \
                            f"but got {input_tensor.dtype}")
                
    for quant_mode, quant_mode_name in zip(quant_mode_list, quant_mode_names):
        if quant_mode is not None:
            if quant_mode not in QUANT_MODE_SUPPORT_LIST:
                raise RuntimeError( \
                    f"{quant_mode_name} should be {[_ for _ in QUANT_MODE_SUPPORT_LIST]}, but got {quant_mode}")