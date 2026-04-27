from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_type, \
torch_dtype_value_to_ge_proto_type, _ge_dtype_to_ge_proto_dtype

Y_DTYPE_SUPPORT_LIST = {
    DataType.DT_FLOAT16,
    DataType.DT_BF16
}

X_DTYPE_WITH_MXFP4_SUPPORT_LIST = {
    DataType.DT_UINT8
}

QUANT_MODE_SUPPORT_LIST = {
    1,
    6
}


WORLD_SIZE_SUPPORT_LIST = [2, 4, 8, 16, 32, 64, 128, 256]
GMM_X_TT_QUANT_MODE = 1
GMM_WEIGHT_TT_QUANT_MODE = 1
GMM_X_MX_QUANT_MODE = 6
GMM_WEIGHT_MX_QUANT_MODE = 6

# group size max value is (2^16 - 1)
GROUP_SIZE_MAX_VALUE = 65535


@register_fx_node_ge_converter(torch.ops.npu.npu_alltoallv_quant_gmm.default)
def convert_npu_alltoallv_quant_gmm(
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
    gmm_x_quant_mode: int = None,
    gmm_weight_quant_mode: int = None,
    mm_x_quant_mode: int = None,
    mm_weight_quant_mode: int = None,
    permute_out_flag: bool = False,
    group_size: Optional[List[int]] = None,
    gmm_x_dtype: int = None,
    gmm_weight_dtype: int = None,
    gmm_x_scale_dtype: int = None,
    gmm_weight_scale_dtype: int = None,
    mm_x_dtype: int = None,
    mm_weight_dtype: int = None,
    mm_x_scale_dtype: int = None,
    mm_weight_scale_dtype: int = None,
    mm_y_dtype: int = None,
    meta_outputs: TensorSpec = None):
    '''
    npu_alltoallv_quant_gmm(Tensor gmm_x, Tensor gmm_weight, Tensor gmm_x_scale, Tensor gmm_weight_scale, 
    str hcom, int ep_world_size, int[] send_counts, int[] recv_counts, int gmm_y_dtype, *, 
    Tensor? send_counts_tensor=None, Tensor? recv_counts_tensor=None, Tensor? mm_x=None, 
    Tensor? mm_weight=None, Tensor? mm_x_scale=None, Tensor? mm_weight_scale=None,  
    int? gmm_x_quant_mode=None, int? gmm_weight_quant_mode=None, int? mm_x_quant_mode=None, 
    int? mm_weight_quant_mode=None, bool permute_out_flag=False, int[]? group_size=None, int? gmm_x_dtype=None, 
    int? gmm_weight_dtype=None, int? gmm_x_scale_dtype=None, int? gmm_weight_scale_dtype=None, 
    int? mm_x_dtype=None, int? mm_weight_dtype=None, int? mm_x_scale_dtype=None, 
    int? mm_weight_scale_dtype=None, int? mm_y_dtype=None) -> (Tensor, Tensor, Tensor)
    '''
    import torch_npu

    dependencies = []
    node_name = None
    trans_gmm_weight = False
    trans_mm_weight = False

    if ep_world_size not in WORLD_SIZE_SUPPORT_LIST:
        raise RuntimeError(f"ep_world_size only supports value in {WORLD_SIZE_SUPPORT_LIST}, but got {ep_world_size}.")

    x_list = [gmm_x, gmm_weight, gmm_x_scale, gmm_weight_scale, mm_x, mm_weight, mm_x_scale, mm_weight_scale]
    x_dtype_list = [gmm_x_dtype, gmm_weight_dtype, gmm_x_scale_dtype, gmm_weight_scale_dtype, mm_x_dtype, \
                    mm_weight_dtype, mm_x_scale_dtype, mm_weight_scale_dtype]
    tensor_names = ["gmm_x", "gmm_weight", "gmm_x_scale", "gmm_weight_scale", "mm_x", "mm_weight", \
                    "mm_x_scale", "mm_weight_scale"]
    dtype_names = ["gmm_x_dtype", "gmm_weight_dtype", "gmm_x_scale_dtype", "gmm_weight_scale_dtype", \
                   "mm_x_dtype", "mm_weight_dtype", "mm_x_scale_dtype", "mm_weight_scale_dtype"]
    quant_mode_list = [gmm_x_quant_mode, gmm_weight_quant_mode, mm_x_quant_mode, mm_weight_quant_mode]
    quant_mode_names = ["gmm_x_quant_mode", "gmm_weight_quant_mode", "mm_x_quant_mode", "mm_weight_quant_mode"]
    check_inputs(x_list, x_dtype_list, quant_mode_list, tensor_names, dtype_names, quant_mode_names)

    gmm_x_quant_mode = 1 if gmm_x_quant_mode is None else gmm_x_quant_mode
    gmm_weight_quant_mode = 1 if gmm_weight_quant_mode is None else gmm_weight_quant_mode
    mm_x_quant_mode = 1 if mm_x_quant_mode is None else mm_x_quant_mode
    mm_weight_quant_mode = 1 if mm_weight_quant_mode is None else mm_weight_quant_mode

    if gmm_x_quant_mode == GMM_X_TT_QUANT_MODE and gmm_weight_quant_mode == GMM_WEIGHT_TT_QUANT_MODE:
        x_list = cast_to_real_type(x_list, x_dtype_list)
        gmm_x, gmm_weight, gmm_x_scale, gmm_weight_scale, mm_x, mm_weight, mm_x_scale, mm_weight_scale = x_list
    elif gmm_x_quant_mode == GMM_X_MX_QUANT_MODE and gmm_weight_quant_mode == GMM_WEIGHT_MX_QUANT_MODE:
        if gmm_weight_scale is not None and gmm_weight_scale_dtype is not None:
            gmm_weight_trans = (gmm_weight.desc.layout == "")
            if gmm_weight_trans is not True:
                perm = [i for i in range(gmm_weight_scale.rank)]
                perm[-2], perm[-3] = perm[-2], perm[-3]
                gmm_weight_scale = ge.Transpose(gmm_weight_scale, perm)
            gmm_weight_scale = ge.Bitcast(gmm_weight_scale, type=torch_dtype_value_to_ge_type(gmm_weight_scale_dtype))
            gmm_weight_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(gmm_weight_scale_dtype)
        if gmm_x is not None and gmm_x_dtype is not None:
            if gmm_weight is not None and gmm_weight_dtype is not None:
                if gmm_x.dtype in X_DTYPE_WITH_MXFP4_SUPPORT_LIST and gmm_weight.dtype in X_DTYPE_WITH_MXFP4_SUPPORT_LIST:
                    factor = 2
                    gmm_weight_trans = gmm_x.symsize[-1] == gmm_weight.symsize[-2]
                    gmm_x_const = ge.Const([1] * (gmm_x.rank - 1) + [factor])
                    gmm_x_shape = ge.Shape(gmm_x)
                    gmm_x_new_shape = ge.Mul(gmm_x_shape, gmm_x_const)
                    gmm_x = ge.Bitcast(gmm_x, type=torch_dtype_value_to_ge_type(gmm_x_dtype))
                    gmm_x.desc.dtype = torch_dtype_value_to_ge_proto_type(gmm_x_dtype)
                    gmm_x = ge.Reshape(gmm_x, gmm_x_new_shape)

                    perm = [i for i in range(gmm_weight.rank)]
                    if gmm_weight_trans:
                        perm[-2], perm[-1] = perm[-1], perm[-2]
                    gmm_weight_const = ge.Const([1] * (gmm_weight.rank - 1) + [factor])
                    if gmm_weight_trans:
                        gmm_weight = ge.Transpose(gmm_weight, perm)
                    gmm_weight_shape = ge.Shape(gmm_weight)
                    gmm_weight_new_shape = ge.Mul(gmm_weight_shape, gmm_weight_const)
                    gmm_weight = ge.Bitcast(gmm_weight, type=torch_dtype_value_to_ge_type(gmm_weight_dtype))
                    gmm_weight.desc.dtype = torch_dtype_value_to_ge_proto_type(gmm_weight_dtype)
                    gmm_weight = ge.Reshape(gmm_weight, gmm_weight_new_shape)
                    if gmm_weight_trans:
                        gmm_weight = ge.Transpose(gmm_weight, perm)
                else:
                    gmm_x = ge.Bitcast(gmm_x, type=torch_dtype_value_to_ge_type(gmm_x_dtype))
                    gmm_x.desc.dtype = torch_dtype_value_to_ge_proto_type(gmm_x_dtype)
                    gmm_weight = ge.Bitcast(gmm_weight, type=torch_dtype_value_to_ge_type(gmm_weight_dtype))
                    gmm_weight.desc.dtype = torch_dtype_value_to_ge_proto_type(gmm_weight_dtype)

        if mm_x is not None and mm_x_dtype is not None:
            if mm_weight is not None and mm_weight_dtype is not None:
                if mm_x.dtype in X_DTYPE_WITH_MXFP4_SUPPORT_LIST and mm_weight.dtype in X_DTYPE_WITH_MXFP4_SUPPORT_LIST:
                    factor = 2
                    mm_weight_trans = mm_x.symsize[-1] == mm_weight.symsize[-2]
                    mm_x_const = ge.Const([1] * (mm_x.rank - 1) + [factor])
                    mm_x_shape = ge.Shape(mm_x)
                    mm_x_new_shape = ge.Mul(mm_x_shape, mm_x_const)
                    mm_x = ge.Bitcast(mm_x, type=torch_dtype_value_to_ge_type(mm_x_dtype))
                    mm_x.desc.dtype = torch_dtype_value_to_ge_proto_type(mm_x_dtype)
                    mm_x = ge.Reshape(mm_x, mm_x_new_shape)

                    perm = [i for i in range(mm_weight.rank)]
                    if mm_weight_trans:
                        perm[-2], perm[-1] = perm[-1], perm[-2]
                    mm_weight_const = ge.Const([1] * (mm_weight.rank - 1) + [factor])
                    if mm_weight_trans:
                        mm_weight = ge.Transpose(mm_weight, perm)
                    mm_weight_shape = ge.Shape(mm_weight)
                    mm_weight_new_shape = ge.Mul(mm_weight_shape, mm_weight_const)
                    mm_weight = ge.Bitcast(mm_weight, type=torch_dtype_value_to_ge_type(mm_weight_dtype))
                    mm_weight.desc.dtype = torch_dtype_value_to_ge_proto_type(mm_weight_dtype)
                    mm_weight = ge.Reshape(mm_weight, mm_weight_new_shape)
                    if mm_weight_trans:
                        mm_weight = ge.Transpose(mm_weight, perm)
                else:
                    mm_x = ge.Bitcast(mm_x, type=torch_dtype_value_to_ge_type(mm_x_dtype))
                    mm_x.desc.dtype = torch_dtype_value_to_ge_proto_type(mm_x_dtype)
                    mm_weight = ge.Bitcast(mm_weight, type=torch_dtype_value_to_ge_type(mm_weight_dtype))
                    mm_weight.desc.dtype = torch_dtype_value_to_ge_proto_type(mm_weight_dtype)
        if gmm_x_scale is not None and gmm_x_scale_dtype is not None:
            gmm_x_scale = ge.Bitcast(gmm_x_scale, type=torch_dtype_value_to_ge_type(gmm_x_scale_dtype))
            gmm_x_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(gmm_x_scale_dtype)
        if mm_x_scale is not None and mm_x_scale_dtype is not None:
            mm_x_scale = ge.Bitcast(mm_x_scale, type=torch_dtype_value_to_ge_type(mm_x_scale_dtype))
            mm_x_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(mm_x_scale_dtype)
        if mm_weight_scale is not None and mm_weight_scale_dtype is not None:
            mm_weight_scale = ge.Bitcast(mm_weight_scale, type=torch_dtype_value_to_ge_type(mm_weight_scale_dtype))
            mm_weight_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(mm_weight_scale_dtype)
    group_sizes = 0
    if group_size is not None and isinstance(group_size, List):
        if (len(group_size) != 3):
            raise RuntimeError("group_size must be a list with 3 elements, actual group_size is " + str(group_size))
        group_m = group_size[0]
        group_n = group_size[1]
        group_k = group_size[2]
        if (group_m > GROUP_SIZE_MAX_VALUE or group_n > GROUP_SIZE_MAX_VALUE or group_k > GROUP_SIZE_MAX_VALUE):
            raise RuntimeError("group_size can't larger than {GROUP_SIZE_MAX_VALUE}, actual group_size is " + str(group_size))
        if (group_m < 0 or group_n < 0 or group_k < 0):
            raise RuntimeError("group_size can't smaller than 0, actual group_size is " + str(group_size))
        group_sizes = (group_m << 32) + (group_n << 16) + group_k
    
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

    (gmm_y, mm_y, permute_out) = ge.AlltoAllvQuantGroupedMatMul(
        gmm_x=gmm_x, 
        gmm_weight=gmm_weight, 
        gmm_x_scale=gmm_x_scale, 
        gmm_weight_scale=gmm_weight_scale, 
        send_counts_tensor=send_counts_tensor, 
        recv_counts_tensor=recv_counts_tensor, 
        mm_x=mm_x, 
        mm_weight=mm_weight, 
        mm_x_scale=mm_x_scale, 
        mm_weight_scale=mm_weight_scale, 
        group=hcom, 
        ep_world_size=ep_world_size, 
        send_counts=send_counts, 
        recv_counts=recv_counts, 
        gmm_x_quant_mode=gmm_x_quant_mode, 
        gmm_weight_quant_mode=gmm_weight_quant_mode, 
        trans_gmm_weight=trans_gmm_weight, 
        trans_mm_weight=trans_mm_weight, 
        permute_out_flag=permute_out_flag, 
        mm_x_quant_mode=mm_x_quant_mode, 
        mm_weight_quant_mode=mm_weight_quant_mode, 
        group_size=group_sizes, 
        y_dtype=output_dtype, 
        mm_dtype=mm_out_dtype, 
        dependencies=dependencies, 
        node_name=node_name)
    
    gmm_y.desc.dtype = _ge_dtype_to_ge_proto_dtype(output_dtype)
    if mm_x is not None and mm_y is not None:
        mm_y.desc.dtype = _ge_dtype_to_ge_proto_dtype(mm_out_dtype)
    if permute_out_flag:
        permute_out_dtype = torch_dtype_value_to_ge_type(gmm_x_dtype) if gmm_x_dtype is not None else gmm_x.dtype
        permute_out.desc.dtype = _ge_dtype_to_ge_proto_dtype(permute_out_dtype)

    return (gmm_y, mm_y, permute_out)


def cast_to_real_type(x_list, x_dtype_list):
    for i in range(len(x_list)):
        if x_dtype_list[i] is not None and x_list[i] is not None:
            x_list[i] = ge.Bitcast(x_list[i], type=torch_dtype_value_to_ge_type(x_dtype_list[i]))
            x_list[i].desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype_list[i])
    return x_list


def check_inputs(tensor_list, dtype_list, quant_mode_list, tensor_names, dtype_names, quant_mode_names):
    """校验输入参数
    
    校验顺序：
    1. 校验必选输入gmm
    2. 校验可选输入mm
    3. 校验是否为标量
    4. 校验量化模式
    5. 数据类型校验
    """
    # ========== Step 1: 校验gmm输入（必选输入） ==========
    gmm_x, gmm_weight, gmm_x_scale, gmm_weight_scale = tensor_list[0], tensor_list[1], tensor_list[2], tensor_list[3]
    if gmm_x is None:
        raise RuntimeError("gmm_x should not be None")
    if gmm_weight is None:
        raise RuntimeError("gmm_weight should not be None")
    if gmm_x_scale is None:
        raise RuntimeError("gmm_x_scale should not be None")
    if gmm_weight_scale is None:
        raise RuntimeError("gmm_weight_scale should not be None")
    
    # ========== Step 2: 校验mm输入（可选输入） ==========   
    mm_all_attrs = [
        tensor_list[4], tensor_list[5], tensor_list[6], tensor_list[7],
        dtype_list[4], dtype_list[5], dtype_list[6], dtype_list[7]
    ]
    mm_attr_names = [
        "mm_x", "mm_weight", "mm_x_scale", "mm_weight_scale",
        "mm_x_dtype", "mm_weight_dtype", "mm_x_scale_dtype", "mm_weight_scale_dtype"
    ]

    mm_all_inputs = [tensor_list[4], tensor_list[5], tensor_list[6], tensor_list[7]]
    has_any_none_mm_input = any(input is None for input in mm_all_inputs)
    if has_any_none_mm_input:
        for attr, name in zip(mm_all_attrs, mm_attr_names):
            if attr is not None:
                raise RuntimeError( \
                    f"When any mm input is None, all mm attributes must be None, but {name} is not None.")
       
    # ========== Step 3: 校验所有tensor的rank不为0 ==========
    for input_tensor, tensor_name in zip(tensor_list, tensor_names):
        if input_tensor is not None and input_tensor.rank == 0:
            raise RuntimeError(f"{tensor_name} dim num should not be 0")
    
    # ========== Step 4: 校验量化模式（仅支持TT和MX） ==========
    for quant_mode, quant_mode_name in zip(quant_mode_list, quant_mode_names):
        if quant_mode is not None:
            if quant_mode not in QUANT_MODE_SUPPORT_LIST:
                raise RuntimeError( \
                    f"{quant_mode_name} should be {[_ for _ in QUANT_MODE_SUPPORT_LIST]}, but got {quant_mode}")
    
    quant_mode_info = []
    for name, mode in zip(quant_mode_names, quant_mode_list):
        quant_mode_info.append(f"{name}={mode}")
    
    non_none_quant_modes = [qm for qm in quant_mode_list if qm is not None]
    if len(non_none_quant_modes) == 0:
        raise RuntimeError( \
            f"Quant mode must be specified, supported modes are {[_ for _ in QUANT_MODE_SUPPORT_LIST]} (1=TT quant, 6=MX quant)")
    
    first_mode = non_none_quant_modes[0]
    for qm in non_none_quant_modes[1:]:
        if qm != first_mode:
            raise RuntimeError( \
                f"All quant modes should be the same (either all 1 or all 6), but got mixed modes: {quant_mode_info}.")
    
    # ========== Step 5: 数据类型校验 （x/weight/scale）==========
    def get_tensor_dtype(tensor):
        if tensor is not None:
            return tensor.dtype
        return None

    def get_dtypeparam_dtype(dtype_param):
        if dtype_param is not None:
            return torch_dtype_value_to_ge_type(dtype_param)
        return None
    
    # 5.1 获取所有x/weight的dtype
    gmm_x_tensor_dtype = get_tensor_dtype(tensor_list[0])
    gmm_weight_tensor_dtype = get_tensor_dtype(tensor_list[1])
    mm_x_tensor_dtype = get_tensor_dtype(tensor_list[4])
    mm_weight_tensor_dtype = get_tensor_dtype(tensor_list[5])
    gmm_x_dtypeparam_dtype = get_dtypeparam_dtype(dtype_list[0])
    gmm_weight_dtypeparam_dtype = get_dtypeparam_dtype(dtype_list[1])
    mm_x_dtypeparam_dtype = get_dtypeparam_dtype(dtype_list[4])
    mm_weight_dtypeparam_dtype = get_dtypeparam_dtype(dtype_list[5])
    all_tensor_dtypes = {
        "gmm_x": gmm_x_tensor_dtype,
        "gmm_weight": gmm_weight_tensor_dtype,
        "mm_x": mm_x_tensor_dtype,
        "mm_weight": mm_weight_tensor_dtype
    }
    all_dtypeparam_dtypes = {
        "gmm_x": gmm_x_dtypeparam_dtype,
        "gmm_weight": gmm_weight_dtypeparam_dtype,
        "mm_x": mm_x_dtypeparam_dtype,
        "mm_weight": mm_weight_dtypeparam_dtype
    }
    
    # 5.2 获取所有scale的dtype
    gmm_x_scale_tensor_dtype = get_tensor_dtype(tensor_list[2])
    gmm_weight_scale_tensor_dtype = get_tensor_dtype(tensor_list[3])
    mm_x_scale_tensor_dtype = get_tensor_dtype(tensor_list[6])
    mm_weight_scale_tensor_dtype = get_tensor_dtype(tensor_list[7])
    gmm_x_scale_dtypeparam_dtype = get_dtypeparam_dtype(dtype_list[2])
    gmm_weight_scale_dtypeparam_dtype = get_dtypeparam_dtype(dtype_list[3])
    mm_x_scale_dtypeparam_dtype = get_dtypeparam_dtype(dtype_list[6])
    mm_weight_scale_dtypeparam_dtype = get_dtypeparam_dtype(dtype_list[7])
    all_scale_tensor_dtypes = {
        "gmm_x_scale": gmm_x_scale_tensor_dtype,
        "gmm_weight_scale": gmm_weight_scale_tensor_dtype,
        "mm_x_scale": mm_x_scale_tensor_dtype,
        "mm_weight_scale": mm_weight_scale_tensor_dtype
    }
    all_scale_dtypeparam_dtypes = {
        "gmm_x_scale": gmm_x_scale_dtypeparam_dtype,
        "gmm_weight_scale": gmm_weight_scale_dtypeparam_dtype,
        "mm_x_scale": mm_x_scale_dtypeparam_dtype,
        "mm_weight_scale": mm_weight_scale_dtypeparam_dtype
    }
    
    # 5.3 根据量化模式校验数据类型
    quant_mode = non_none_quant_modes[0]
    if quant_mode == 1:
        # 5.3.1 校验TT量化：x和weight输入数据类型为hifloat8，scale为float32（hifloat8需要uint8包装）
        # 校验x/weight
        for name, dtype in all_tensor_dtypes.items():
            if dtype is not None and dtype != DataType.DT_UINT8 and all_dtypeparam_dtypes.get(name) != DataType.DT_HIFLOAT8:
                raise RuntimeError( \
                    f"When quant mode is 1 (TT quant), all x/weight must be UINT8 (for HIFLOAT8), \
                        and the {name}_dtype must be specified as HIFLOAT8, but {name} is {torch_dtype_value_to_ge_type(dtype)}, \
                        {name}_dtype is {all_dtypeparam_dtypes.get(name)}")
        # 校验scale      
        for name, dtype in all_scale_tensor_dtypes.items():
            if dtype is not None and dtype != DataType.DT_FLOAT:
                raise RuntimeError( \
                    f"When quant mode is 1 (TT quant), all scales must be FLOAT, but {name} is {dtype}.")
            if dtype is not None and all_scale_dtypeparam_dtypes.get(name) is not None and dtype != all_scale_dtypeparam_dtypes.get(name):
                raise RuntimeError( \
                    f"When quant mode is 1 (TT quant), {name}_dtype is not necessary, it can be None. \
                        However, if {name}_dtype is not None, the {name}_dtype must be same as the {name} tensor's dtype, \
                        but now the tensor's dtype is: {dtype}, the {name}_dtype is {all_dtypeparam_dtypes.get(name)}.")                     
    elif quant_mode == 6:
        # 5.3.2 校验MX量化：x和weight输入数据类型为fp8_e5m2/fp8_e4m3fn/fp4_e2m1，scale为fp8_e8m0（fp4_e2m1和fp8_e8m0需要uint8包装）      
        FP8_SUPPORT_DTYPES_LIST = {DataType.DT_FLOAT8_E5M2, DataType.DT_FLOAT8_E4M3FN}       
        # 校验x/weight
        uint8_tensors = []
        fp8_tensors = []
        for name, dtype in all_tensor_dtypes.items():
            if dtype is not None and dtype == DataType.DT_UINT8:
                uint8_tensors.append(name)
            elif dtype in FP8_SUPPORT_DTYPES_LIST:
                fp8_tensors.append(name)
            elif dtype is None:
                pass
            else:
                raise RuntimeError( \
                    f"When quant mode is 6 (MX quant), tensor {name} dtype must be UINT8 (for FP4_E2M1) or FP8 (FP8_E5M2 or FP8_E4M3FN), but got {dtype}.")         
        # FP4场景校验：tensor为uint8时，dtype必须为fp4_e2m1
        for name in uint8_tensors:
            dtype_param = all_dtypeparam_dtypes.get(name)
            if dtype_param is None:
                raise RuntimeError( \
                    f"When quant mode is 6 (MX quant) and tensor {name} is UINT8 (for FP4_E2M1), the {name}_dtype must be specified as FP4_E2M1, but got None.")
            if dtype_param != DataType.DT_FLOAT4_E2M1:
                raise RuntimeError( \
                    f"When quant mode is 6 (MX quant) and tensor {name} is UINT8 (for FP4_E2M1), the {name}_dtype must be specified as FLOAT4_E2M1, but got {dtype_param}.")       
        # 一致性校验：如果有任何输入为fp4_e2m1，则所有输入都应该是fp4_e2m1
        has_fp4 = any(dtype == DataType.DT_FLOAT4_E2M1 for dtype in all_dtypeparam_dtypes.values() if dtype is not None)
        if has_fp4:
            for name, dtype in all_dtypeparam_dtypes.items():
                if dtype is not None and dtype != DataType.DT_FLOAT4_E2M1:
                    raise RuntimeError( \
                        f"When any x/weight is FLOAT4_E2M1 (MX quant), all x/weight must be FLOAT4_E2M1, but {name} is {dtype}.")
        # FP8场景校验：tensor为fp8时，dtype应该为None或与tensor一致
        for name in fp8_tensors:
            dtype_param = all_dtypeparam_dtypes.get(name)
            if dtype_param is not None and dtype_param != all_tensor_dtypes.get(name):
                raise RuntimeError( \
                    f"When quant mode is 6 (MX quant) and tensor {name} is FP8, {name}_dtype is not necessary, it can be None. \
                        However, if {name}_dtype is not None, the {name}_dtype must be same as the {name} tensor's dtype, \
                        but now the tensor's dtype is: {all_tensor_dtypes.get(name)}, the {name}_dtype is {dtype_param}.")                   
        # scale校验
        for name, dtype in all_scale_tensor_dtypes.items():            
            if dtype is not None and dtype != DataType.DT_UINT8 and all_scale_dtypeparam_dtypes.get(name) != DataType.DT_FLOAT8_E8M0:
                raise RuntimeError( \
                    f"When quant mode is 6 (MX quant), all scale must be UINT8 (for FP8_E8M0), \
                        and the {name}_dtype must be specified as FP8_E8M0, but {name} is {torch_dtype_value_to_ge_type(dtype)}, \
                        {name}_dtype is {all_scale_dtypeparam_dtypes.get(name)}")            
    
    # 5.4 校验gmm和mm的x/weight dtype一致性
    if not has_any_none_mm_input:
        if all_tensor_dtypes["mm_x"] != all_tensor_dtypes["gmm_x"]:
            raise RuntimeError( \
                f"gmm_x and mm_x must have the same dtype, but gmm_x is {all_tensor_dtypes['gmm_x']} and mm_x is {all_tensor_dtypes['mm_x']}.")
        if all_tensor_dtypes["mm_weight"] != all_tensor_dtypes["gmm_weight"]:
            raise RuntimeError( \
                f"gmm_weight and mm_weight must have the same dtype, but gmm_weight is {all_tensor_dtypes['gmm_weight']} and mm_weight is {all_tensor_dtypes['mm_weight']}.")
        if all_scale_tensor_dtypes["mm_x_scale"] != all_scale_tensor_dtypes["gmm_x_scale"]:
            raise RuntimeError( \
                f"gmm_x_scale and mm_x_scale must have the same dtype, \
                    but gmm_x_scale is {all_scale_tensor_dtypes['gmm_x_scale']} and mm_x_scale is {all_scale_tensor_dtypes['mm_x_scale']}.")
        if all_scale_tensor_dtypes["mm_weight_scale"] != all_scale_tensor_dtypes["gmm_weight_scale"]:
            raise RuntimeError( \
                f"gmm_weight_scale and mm_weight_scale must have the same dtype, \
                    but gmm_weight_scale is {all_scale_tensor_dtypes['gmm_weight_scale']} and mm_weight_scale is {all_scale_tensor_dtypes['mm_weight_scale']}.")