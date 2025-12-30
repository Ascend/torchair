from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair._utils.check_platform import is_arch35
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_proto_type, torch_dtype_value_to_ge_type


TORCH_DTYPE_MAP = {
    torch.float16: 5,
    torch.bfloat16: 15,
    torch.float32: 6,
    torch.float8_e5m2: 23,
    torch.float8_e4m3fn: 24,
    torch.bits8: 21,
    torch.int8: 1,
    torch.int32: 3,
}

INT4_NUMS_IN_INT32 = 8


def fill_empty_tensorlist(input_data, desired_dtype):
    if not input_data:
        return [ge.Fill([0], ge.Cast(0., dst_type=desired_dtype))]
    else:
        return input_data


def fp8_quant_mode(intput_x_dtype, x_dtype, weight_dtype, scale):
    if scale is None:
        return False
    if intput_x_dtype == DataType.DT_FLOAT8_E4M3FN or intput_x_dtype == DataType.DT_FLOAT8_E5M2:
        return True
    if (x_dtype is not None and torch_dtype_value_to_ge_type(x_dtype) == DataType.DT_HIFLOAT8) and \
       (weight_dtype is not None and torch_dtype_value_to_ge_type(weight_dtype) == DataType.DT_HIFLOAT8):
        return True
    return False


def fp4_quant_mode(x_dtype, weight_dtype, scale):
    if scale is None:
        return False
    if ((x_dtype is not None and torch_dtype_value_to_ge_type(x_dtype) == DataType.DT_FLOAT4_E2M1) and \
       (weight_dtype is not None and torch_dtype_value_to_ge_type(weight_dtype) == DataType.DT_FLOAT4_E2M1)) or \
       ((x_dtype is not None and torch_dtype_value_to_ge_type(x_dtype) == DataType.DT_FLOAT4_E1M2) and \
       (weight_dtype is not None and torch_dtype_value_to_ge_type(weight_dtype) == DataType.DT_FLOAT4_E1M2)):
        return True
    return False


def a16_weight_quant_mode(x_dtype, weight_dtype):
    return (x_dtype == DataType.DT_FLOAT16 or x_dtype == DataType.DT_BF16) and \
        (weight_dtype == DataType.DT_INT8 or weight_dtype == DataType.DT_FLOAT or weight_dtype == DataType.DT_INT32)


def arch35_a16w4_weight_quant_mode(x_dtype, weight_dtype):
    return is_arch35() and weight_dtype == DataType.DT_INT32 and \
        (x_dtype == DataType.DT_FLOAT16 or x_dtype == DataType.DT_BF16)


def convert_tensorlist_to_int4(input_data: List[Tensor], trans: bool):
    input_dtype = input_data[0].dtype
    w_list = []
    if input_dtype == DataType.DT_INT32:
        for w_item in input_data:
            perm = [i for i in range(w_item.rank)]
            perm[-1], perm[-2] = perm[-2], perm[-1]
            if trans:
                w_item = ge.Transpose(w_item, perm)
            from torch_npu.npu.utils import _is_gte_cann_version
            if _is_gte_cann_version("8.5.0"):
                new_w = ge.Bitcast(w_item, type=DataType.DT_INT4, keep_dim=True)
            else:
                const_w = ge.Const([1] * (w_item.rank - 1) + [8])
                shape_w = ge.Shape(w_item)
                shape_w = ge.Mul(shape_w, const_w)
                new_w = ge.Bitcast(w_item, type=DataType.DT_INT4)
                new_w = ge.Reshape(new_w, shape_w)

            if trans:
                new_w = ge.Transpose(new_w, perm)
            w_list.append(new_w)
    else:
        w_list = input_data
    return w_list


def convert_tensorlist_to_int4_arch35(input_data: List[Tensor], trans: bool):
    w_list = []
    for w_item in input_data:
        const_w = ge.Const([1] * (w_item.rank - 1) + [INT4_NUMS_IN_INT32])
        perm = [i for i in range(w_item.rank)]
        perm[-1], perm[-2] = perm[-2], perm[-1]
        if trans:
            w_item = ge.Transpose(w_item, perm)
        shape_w = ge.Shape(w_item)
        shape_w = ge.Mul(shape_w, const_w)
        new_w = ge.Bitcast(w_item, type=DataType.DT_INT4)
        new_w = ge.Reshape(new_w, shape_w)
        if trans:
            new_w = ge.Transpose(new_w, perm)
        w_list.append(new_w)
    return w_list


def convert_tensorlist_to_mxfp4_item(input_data: Tensor, x_dtype, trans):
    shape_multiples = 2
    x_ge_dtype = 0

    if x_dtype is not None:
        x_ge_dtype = torch_dtype_value_to_ge_type(x_dtype)
    const_x = ge.Const([1] * (input_data.rank - 1) + [shape_multiples])
    perm = [i for i in range(input_data.rank)]
    perm[-1], perm[-2] = perm[-2], perm[-1]
    if trans:
        input_data = ge.Transpose(input_data, perm)
    shape_x = ge.Shape(input_data)
    shape_x = ge.Mul(shape_x, const_x)
    input_data = ge.Bitcast(input_data, type=x_ge_dtype)
    input_data = ge.Reshape(input_data, shape_x)
    if trans:
        input_data = ge.Transpose(input_data, perm)
    return input_data


def convert_tensorlist_to_mxfp4(x: List[Tensor], weight: List[Tensor], x_dtype, weight_dtype):
    x_list = []
    w_list = []
    for x_item in x:
        new_x = convert_tensorlist_to_mxfp4_item(x_item, x_dtype, False)
        x_list.append(new_x)
    for w_item in weight:
        # x只有一个x_item
        trans_weight = x_item.symsize[-1] == w_item.symsize[-2]
        new_w = convert_tensorlist_to_mxfp4_item(w_item, weight_dtype, trans_weight)
        w_list.append(new_w)
    return x_list, w_list


def convert_scale_tensorlist(scales: List[Tensor]):
    scale_type = scales[0].dtype
    output = []
    if scale_type == DataType.DT_INT64:
        for scale in scales:
            item = ge.Cast(scale, dst_type=DataType.DT_UINT64)
            output.append(item)
    else:
        output = scales
    return output


def convert_ydtype_in_a8(output_dtype: Optional[int] = None):
    y_dtype = -1
    if output_dtype == TORCH_DTYPE_MAP[torch.float16]:
        y_dtype = 0
    elif output_dtype == TORCH_DTYPE_MAP[torch.bfloat16]:
        y_dtype = 1
    elif output_dtype == TORCH_DTYPE_MAP[torch.int32]:
        y_dtype = 2
    elif output_dtype == TORCH_DTYPE_MAP[torch.float32]:
        if not is_arch35():
            raise ValueError("output_dtype does not support float32 in versions prior to A5.")
        y_dtype = 3
    elif output_dtype == TORCH_DTYPE_MAP[torch.int8]:
        raise ValueError("output_dtype not support int8 yet for graph mode")
    else:
        raise ValueError(f"output_dtype should be float16, bfloat16 or int32, "
                            f"otherwise it should be None, but got {output_dtype}")
    return y_dtype


def convert_ydtype_in_a4(output_dtype: Optional[int] = None):
    y_dtype = -1
    if output_dtype == TORCH_DTYPE_MAP[torch.float16]:
        y_dtype = 0
    elif output_dtype == TORCH_DTYPE_MAP[torch.bfloat16]:
        y_dtype = 1
    else:
        raise ValueError(f"output_dtype should be float16, bfloat16, "
                            f"otherwise it should be None, but got {output_dtype}")
    return y_dtype


def conveter_npu_npu_grouped_matmul(
    x: List[Tensor],
    weight: List[Tensor],
    *,
    bias: Optional[List[Tensor]] = None,
    scale: Optional[List[Tensor]] = None,
    offset: Optional[List[Tensor]] = None,
    antiquant_scale: Optional[List[Tensor]] = None,
    antiquant_offset: Optional[List[Tensor]] = None,
    per_token_scale: Optional[List[Tensor]] = None,
    group_list: Optional[Union[List[int], Tensor]] = None,
    activation_input: Optional[List[Tensor]] = None,
    activation_quant_scale: Optional[List[Tensor]] = None,
    activation_quant_offset: Optional[List[Tensor]] = None,
    split_item: Optional[int] = 0,
    group_type: Optional[int] = -1,
    group_list_type: Optional[int] = 0,
    act_type: Optional[int] = 0,
    tuning_config: Optional[List[int]] = None,
    output_dtype: Optional[int] = None,
    x_dtype: Optional[int] = None,
    weight_dtype: Optional[int] = None,
    scale_dtype: Optional[int] = None,
    per_token_scale_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: npu::npu_grouped_matmul(x, weight, *, bias=None, scale=None, offset=None, antiquant_scale=None,
            antiquant_offset=None, per_token_scale=None, group_list=None,
            activation_input=None, activation_quant_offset=None, activation_quant_offset=None,
            split_item=0, group_type=-1, group_list_type=0, act_type=0,
            tuning_config=[0], output_dtype=None, x_dtype=None, weight_dtype=None, scale_dtype=None, per_token_scale_dtype=None) -> Tensor[]
    """

    import torch_npu

    tuning_config = tuning_config or [0]
    input_x_dtype = x[0].dtype
    w_dtype = weight[0].dtype

    if fp8_quant_mode(input_x_dtype, x_dtype, weight_dtype, scale):
        if bias is None:
            bias = fill_empty_tensorlist(bias, DataType.DT_FLOAT)
    elif fp4_quant_mode(x_dtype, weight_dtype, scale):
        if bias is None:
            bias = fill_empty_tensorlist(bias, DataType.DT_FLOAT)
    else:
        if input_x_dtype == DataType.DT_BF16:
            bias = fill_empty_tensorlist(bias, DataType.DT_FLOAT)
        elif input_x_dtype == DataType.DT_INT8:
            if w_dtype == DataType.DT_INT32:
                bias = fill_empty_tensorlist(bias, DataType.DT_FLOAT)
            else:
                bias = fill_empty_tensorlist(bias, DataType.DT_INT32)
        elif input_x_dtype == DataType.DT_FLOAT8_E4M3FN and w_dtype == DataType.DT_FLOAT:
            ge_out_type = torch_dtype_value_to_ge_type(output_dtype)
            bias = fill_empty_tensorlist(bias, ge_out_type)
        else:
            bias = fill_empty_tensorlist(bias, input_x_dtype)

    scale = fill_empty_tensorlist(scale, DataType.DT_UINT64)
    scale = convert_scale_tensorlist(scale)
    offset = fill_empty_tensorlist(offset, DataType.DT_FLOAT)

    if a16_weight_quant_mode(input_x_dtype, w_dtype):
        antiquant_scale = fill_empty_tensorlist(antiquant_scale, input_x_dtype)
        antiquant_offset = fill_empty_tensorlist(antiquant_offset, input_x_dtype)
    elif input_x_dtype == DataType.DT_FLOAT8_E4M3FN and w_dtype == DataType.DT_FLOAT:
        ge_out_type = torch_dtype_value_to_ge_type(output_dtype)
        antiquant_scale = fill_empty_tensorlist(antiquant_scale, ge_out_type)
        antiquant_offset = fill_empty_tensorlist(antiquant_offset, ge_out_type)
    else:
        antiquant_scale = fill_empty_tensorlist(antiquant_scale, DataType.DT_FLOAT16)
        antiquant_offset = fill_empty_tensorlist(antiquant_offset, DataType.DT_FLOAT16)

    if antiquant_scale[0].dtype == DataType.DT_UINT8:
        antiquant_scale[0] = ge.Bitcast(antiquant_scale[0], type=DataType.DT_FLOAT8_E8M0)
        antiquant_scale[0].desc.dtype = DataType.DT_FLOAT8_E8M0

    y_dtype = -1
    if is_arch35():
        if output_dtype is not None:
            y_dtype = convert_ydtype_in_a8(output_dtype)
    else:
        if output_dtype is not None and x[0].dtype == DataType.DT_INT8:
            y_dtype = convert_ydtype_in_a8(output_dtype)
        elif output_dtype is not None and x[0].dtype == DataType.DT_INT32:
            y_dtype = convert_ydtype_in_a4(output_dtype)

    per_token_scale = per_token_scale[0] if per_token_scale is not None and len(per_token_scale) else None

    if group_list is not None and not isinstance(group_list, Tensor):
        group_list = dtype_promote(group_list, target_dtype=torch.int64)

    x_list = []
    w_list = []
    if weight[0].dtype == DataType.DT_INT32 and not is_arch35():
        x_list = x
        w_list = convert_tensorlist_to_int4(weight, x[0].symsize[-1] == weight[0].symsize[-2] * INT4_NUMS_IN_INT32)
    elif arch35_a16w4_weight_quant_mode(input_x_dtype, w_dtype):
        x_list = x
        w_list = convert_tensorlist_to_int4_arch35(weight, 
                                                   x[0].symsize[-1] == weight[0].symsize[-2] * INT4_NUMS_IN_INT32)
    elif x_dtype is not None and weight_dtype is not None and \
         (x_dtype == torch_npu.float4_e2m1fn_x2 or x_dtype == torch_npu.float4_e1m2fn_x2) and \
         (weight_dtype == torch_npu.float4_e2m1fn_x2 or weight_dtype == torch_npu.float4_e1m2fn_x2):
        x_list, w_list = convert_tensorlist_to_mxfp4(x, weight, x_dtype, weight_dtype)
    else:
        x_list = x
        w_list = weight

    if x_dtype is not None:
        if x_dtype != torch_npu.float4_e2m1fn_x2 and x_dtype != torch_npu.float4_e1m2fn_x2:
            x_list[0] = ge.Bitcast(x_list[0], type=torch_dtype_value_to_ge_type(x_dtype))
        x_list[0].desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype)

    if weight_dtype is not None:
        if weight_dtype != torch_npu.float4_e2m1fn_x2 and weight_dtype != torch_npu.float4_e1m2fn_x2:
            w_list[0] = ge.Bitcast(w_list[0], type=torch_dtype_value_to_ge_type(weight_dtype))
        w_list[0].desc.dtype = torch_dtype_value_to_ge_proto_type(weight_dtype)

    if scale_dtype is not None:
        scale[0] = ge.Bitcast(scale[0], type=torch_dtype_value_to_ge_type(scale_dtype))
        scale[0].desc.dtype = torch_dtype_value_to_ge_proto_type(scale_dtype)

    if per_token_scale_dtype is not None:
        per_token_scale = ge.Bitcast(per_token_scale, type=torch_dtype_value_to_ge_type(per_token_scale_dtype))
        per_token_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(per_token_scale_dtype)

    return ge.GroupedMatmul(x_list, w_list, bias, scale, offset, antiquant_scale, antiquant_offset, group_list,
                            per_token_scale, split_item=split_item, dtype=y_dtype, transpose_weight=False,
                            transpose_x=False, group_type=group_type, group_list_type=group_list_type,
                            act_type=act_type, tuning_config=tuning_config)


gmm_reg = register_fx_node_ge_converter(torch.ops.npu.npu_grouped_matmul.default)(conveter_npu_npu_grouped_matmul)
gmm_List_reg = register_fx_node_ge_converter(torch.ops.npu.npu_grouped_matmul.List)(conveter_npu_npu_grouped_matmul)
