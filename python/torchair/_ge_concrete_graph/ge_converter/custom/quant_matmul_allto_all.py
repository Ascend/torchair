from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_type, \
torch_dtype_value_to_ge_proto_type, _ge_dtype_to_ge_proto_dtype

WORLD_SIZE_SUPPORT_LIST = [2, 4, 8, 16]

# kc quant mode
X_DTYPE_WITH_KC_SUPPORT_LIST = {
    DataType.DT_FLOAT8_E4M3FN,
    DataType.DT_FLOAT8_E5M2
}

# kc quant mode, x_scale tensor input type is float32
X_SCALE_DTYPE_WITH_KC_SUPPORT_LIST = {
    DataType.DT_FLOAT
}

# mxfp8 quant mode
X_DTYPE_WITH_MXFP8_SUPPORT_LIST = {
    DataType.DT_FLOAT8_E4M3FN,
    DataType.DT_FLOAT8_E5M2
}
# mxfp4 quant mode: x tensor input type is uint8, x_dtype is float4_e2m1
X_DTYPE_WITH_MXFP4_SUPPORT_LIST = {
    DataType.DT_UINT8
}

# mx quant mode, x_scale tensor input type is uint8, x_scale_dtype is float8_e8m0
X_SCALE_DTYPE_WITH_MX_SUPPORT_LIST = {
    DataType.DT_UINT8
}

Y_DTYPE_SUPPORT_LIST = {
    DataType.DT_FLOAT,
    DataType.DT_FLOAT16,
    DataType.DT_BF16
}

# torch_npu.float value
TORCH_NPU_FLOAT_VALUE = 6

# x1 & x2 kc quant mode value
X1_KC_QUANT_MODE = 3
X2_KC_QUANT_MODE = 2
# x1 & x2 mx quant mode value
X1_MX_QUANT_MODE = 6
X2_MX_QUANT_MODE = 6

# group size max value is (2^16 - 1)
GROUP_SIZE_MAX_VALUE = 65535 


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
    comm_quant_dtype: int = -1,
    x1_dtype: int = None,
    x2_dtype: int = None,
    x1_scale_dtype: int = None,
    x2_scale_dtype: int = None,
    output_scale_dtype: int = None,
    comm_scale_dtype: int = None,
    y_dtype: int = None,
    meta_outputs: TensorSpec = None
):
    '''
    NB: npu::npu_quant_matmul_all_to_all(Tensor x1, Tensor x2, str hcom, int world_size, Tensor? bias=None, Tensor? x1_scale=None, Tensor? x2_scale=None,
    Tensor? common_scale=None, Tensor? x1_offset=None, Tensor? x1_offset=None, int? x1_quant_mode=0, int? x2_quant_mode=0, int? common_quant_mode=0,
    int[]? group_sizes=None, int[]? all2all_axes=[-1, -2]), int? comm_quant_dtype=28, int? x1_dtype=None, int? x2_dtype=None, int? x1_scale_dtype=None,
    int? x2_scale_dtype=None, int? output_scale_dtype=None, int? comm_scale_dtype=None, int? y_dtype=28) -> Tensor
    '''
    import torch_npu

    # transpose_x1 and transpose_x2 is default set to False
    transpose_x1 = False
    transpose_x2 = False
    common_quant_mode = 0
    comm_quant_dtype = -1

    if all2all_axes is None:
        all2all_axes = [-1, -2]

    if world_size not in WORLD_SIZE_SUPPORT_LIST:
        raise RuntimeError(f"The world_size only supports value in {WORLD_SIZE_SUPPORT_LIST}, but got {world_size}.")

    if bias is not None:
        if bias.dtype != DataType.DT_FLOAT:
            raise AssertionError(f"If bias is not none, the data type of bias:{ge_type_to_torch_type(bias.dtype)} should be float32.")

    if x1_scale is None:
        raise RuntimeError(f"The x1_scale cannot be none.")
    if x2_scale is None:
        raise RuntimeError(f"The x2_scale cannot be none.")

    if x1_quant_mode == X1_KC_QUANT_MODE and x2_quant_mode == X2_KC_QUANT_MODE:
        if x1.dtype not in X_DTYPE_WITH_KC_SUPPORT_LIST:
            raise AssertionError(f"In kc quant scene: x1_quant_mode=3 and x2_quant_mode=2, "
                                 f"type of x1:{ge_type_to_torch_type(x1.dtype)} is error, "
                                 f"x1 should be {[ge_type_to_torch_type(d) for d in X_DTYPE_WITH_KC_SUPPORT_LIST]}.")
        if x2.dtype not in X_DTYPE_WITH_KC_SUPPORT_LIST:
            raise AssertionError(f"In kc quant scene: x1_quant_mode=3 and x2_quant_mode=2, "
                                 f"type of x2:{ge_type_to_torch_type(x2.dtype)} is error, "
                                 f"x2 should be {[ge_type_to_torch_type(d) for d in X_DTYPE_WITH_KC_SUPPORT_LIST]}.")

        if x1_scale.dtype not in X_SCALE_DTYPE_WITH_KC_SUPPORT_LIST:
            raise AssertionError(f"In kc quant scene: x1_quant_mode=3 and x2_quant_mode=2, "
                                 f"tensor type of x1_scale: {ge_type_to_torch_type(x1_scale.dtype)} is error, "
                                 f"x1_scale should be {[ge_type_to_torch_type(d) for d in X_SCALE_DTYPE_WITH_KC_SUPPORT_LIST]}")

        if x2_scale.dtype not in X_SCALE_DTYPE_WITH_KC_SUPPORT_LIST:
            raise AssertionError(f"In kc quant scene: x1_quant_mode=3 and x2_quant_mode=2, "
                                 f"tensor type of x2_scale: {ge_type_to_torch_type(x2_scale.dtype)} is error, "
                                 f"x2_scale should be {[ge_type_to_torch_type(d) for d in X_SCALE_DTYPE_WITH_KC_SUPPORT_LIST]}")

    elif x1_quant_mode == X1_MX_QUANT_MODE and x2_quant_mode == X2_MX_QUANT_MODE:
        if x1.dtype in X_DTYPE_WITH_MXFP8_SUPPORT_LIST and x2.dtype in X_DTYPE_WITH_MXFP8_SUPPORT_LIST:
            # float8_e4m3 and float8_e5m2
            pass
        elif x1.dtype in X_DTYPE_WITH_MXFP4_SUPPORT_LIST and x2.dtype in X_DTYPE_WITH_MXFP4_SUPPORT_LIST:
            # float4_e2m1
            if x1_dtype is None:
                # torch_npu.float4_e2m1fn_x2
                raise AssertionError(f"In mx quant scene: x1_quant_mode=6 and x2_quant_mode=6, "
                                     f"x1_dtype cannot be none.")
            if x2_dtype is None:
                # torch_npu.float4_e2m1fn_x2
                raise AssertionError(f"In mx quant scene: x1_quant_mode=6 and x2_quant_mode=6, "
                                     f"x2_dtype cannot be none.")
            x1_ge_dtype = torch_dtype_value_to_ge_type(x1_dtype)
            x2_ge_dtype = torch_dtype_value_to_ge_type(x2_dtype)
            if not (x1_ge_dtype == DataType.DT_FLOAT4_E2M1 and x2_ge_dtype == DataType.DT_FLOAT4_E2M1):
                raise AssertionError(f"In mx quant scene: x1_quant_mode=6 and x2_quant_mode=6, "
                                     f"x1_dtype and x2_dtype must be float4_e2m1, "
                                     f"but actual x1_dtype is: {x1_dtype}, x2_dtype is: {x2_dtype}.")

            # bitcast接口把uint8强转成float4_e2m1类型，会产生扩维。为了避免这种情况，我们需要将最后一维乘一个系数2
            if (x1.rank < 2):
                raise RuntimeError("In mx quant scene: x1_quant_mode=6 and x2_quant_mode=6, "
                                   "x1 dimension cannot be less than 2, "
                                   "actual x1 dimension is: " + str(x1.rank) + ".")
            if (x2.rank < 2):
                raise RuntimeError("In mx quant scene: x1_quant_mode=6 and x2_quant_mode=6, "
                                   "x2 dimension cannot be less than 2, "
                                   "actual x2 dimension is: " + str(x2.rank) + ".")
            factor = 2
            trans_x2 = x1.symsize[-1] == x2.symsize[-2]

            # 如果x1_shape是2维，得到一个[1, 2]的列表，如果是3维，得到一个[1, 1, 2]的列表
            x1_const = ge.Const([1] * (x1.rank - 1) + [factor])
            x1_shape = ge.Shape(x1)
            x1_new_shape = ge.Mul(x1_shape, x1_const)
            x1 = ge.Bitcast(x1, type=x1_ge_dtype)
            x1.desc.dtype = torch_dtype_value_to_ge_proto_type(x1_dtype)
            x1 = ge.Reshape(x1, x1_new_shape)

            perm = [i for i in range(x2.rank)]
            perm[-2], perm[-1] = perm[-1], perm[-2]
            x2_const = ge.Const([1] * (x2.rank - 1) + [factor])
            if trans_x2:
                x2 = ge.Transpose(x2, perm)
            x2_shape = ge.Shape(x2)
            x2_new_shape = ge.Mul(x2_shape, x2_const)
            x2 = ge.Bitcast(x2, type=x2_ge_dtype)
            x2.desc.dtype = torch_dtype_value_to_ge_proto_type(x2_dtype)
            x2 = ge.Reshape(x2, x2_new_shape)
            if trans_x2:
                x2 = ge.Transpose(x2, perm)
        else:
            raise AssertionError(f"In mx quant scene: x1_quant_mode=6 and x2_quant_mode=6, "
                                 f"tensor type of x1:{ge_type_to_torch_type(x1.dtype)} and x2:{ge_type_to_torch_type(x2.dtype)} is error, "
                                 f"in mxfp8 quant mode, x1 and x2 should be {[ge_type_to_torch_type(d) for d in X_DTYPE_WITH_MXFP8_SUPPORT_LIST]}, "
                                 f"in mxfp4 quant mode, x1 and x2 should be {[ge_type_to_torch_type(d) for d in X_DTYPE_WITH_MXFP4_SUPPORT_LIST]}")

        if x1_scale.dtype in X_SCALE_DTYPE_WITH_MX_SUPPORT_LIST and x2_scale.dtype in X_SCALE_DTYPE_WITH_MX_SUPPORT_LIST:
            if x1_scale_dtype is None:
                # default torch type is 44, to ge type 37
                raise AssertionError(f"In mx quant scene: x1_quant_mode=6 and x2_quant_mode=6, "
                                     f"x1_scale_dtype cannot be none.")
            if x2_scale_dtype is None:
                # default torch type is 44, to ge type 37
                raise AssertionError(f"In mx quant scene: x1_quant_mode=6 and x2_quant_mode=6, "
                                     f"x2_scale_dtype cannot be none.")

            x1_scale_ge_type = torch_dtype_value_to_ge_type(x1_scale_dtype)
            x2_scale_ge_type = torch_dtype_value_to_ge_type(x2_scale_dtype)
            if x1_scale_ge_type != DataType.DT_FLOAT8_E8M0 or x2_scale_ge_type != DataType.DT_FLOAT8_E8M0:
                raise AssertionError(f"In mx quant scene: x1_quant_mode=6 and x2_quant_mode=6, "
                                    f"the x1_scale and x2_scale data type should be float8_e8m0, "
                                    f"but actual x1_scale_dtype is:{ge_type_to_torch_type(x1_scale_ge_type)}, "
                                    f"x2_scale_dtype is:{ge_type_to_torch_type(x2_scale_ge_type)}.")

            x1_scale = ge.Bitcast(x1_scale, type=x1_scale_ge_type)
            x1_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(x1_scale_dtype)
            x2_scale = ge.Bitcast(x2_scale, type=x2_scale_ge_type)
            x2_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(x2_scale_dtype)
        else:
            raise AssertionError(f"In mx quant scene: x1_quant_mode=6 and x2_quant_mode=6, "
                                 f"tensor type of x1_scale: {ge_type_to_torch_type(x1_scale.dtype)} and "
                                 f"tensor type of x2_scale: {ge_type_to_torch_type(x2_scale.dtype)} is error, "
                                 f"x1_scale and x2_scale should be {[ge_type_to_torch_type(d) for d in X_SCALE_DTYPE_WITH_MX_SUPPORT_LIST]}")
    else:
        raise RuntimeError(f"x1 and x2 quant_mode only support 3(dynamic pertoken quant) and 2(perchannel quant) or 6(mx quant) and 6(mx quant), "
                           f"but got x1_quant_mode is:{x1_quant_mode} and x2_quant_mode is:{x2_quant_mode}.")

    output_dtype = None
    if y_dtype is None:
        output_dtype = DataType.DT_FLOAT
    else:
        output_dtype = torch_dtype_value_to_ge_type(y_dtype)
    if output_dtype not in Y_DTYPE_SUPPORT_LIST:
        raise RuntimeError(f"y_dtype should be {[ge_type_to_torch_type(d) for d in Y_DTYPE_SUPPORT_LIST]}, but got {y_dtype}.")

    group_size = 0
    if group_sizes is not None and isinstance(group_sizes, List):
        if(len(group_sizes) != 3):
            raise RuntimeError("group_size must be a list with 3 elements, actual group_sizes is " + str(group_sizes))
        group_m = group_sizes[0]
        group_n = group_sizes[1]
        group_k = group_sizes[2]
        if (group_m > GROUP_SIZE_MAX_VALUE or group_n > GROUP_SIZE_MAX_VALUE or group_k > GROUP_SIZE_MAX_VALUE):
            raise RuntimeError("group_size can't large than {GROUP_SIZE_MAX_VALUE}, actual group_sizes is " + str(group_sizes))
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
