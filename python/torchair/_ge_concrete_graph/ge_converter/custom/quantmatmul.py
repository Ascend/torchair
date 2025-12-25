from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair._utils.check_platform import is_arch35
from torchair.ge._ge_graph import DataType, torch_dtype_value_to_ge_proto_type, torch_dtype_value_to_ge_type


@register_fx_node_ge_converter(torch.ops.npu.npu_quant_matmul.default)
def conveter_npu_npu_quant_matmul(
    x1: Tensor,
    x2: Tensor,
    scale: Tensor,
    *,
    offset: Optional[Tensor] = None,
    pertoken_scale: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    output_dtype: int = None,
    x1_dtype: int = None,
    x2_dtype: int = None,
    pertoken_scale_dtype: int = None,
    scale_dtype: int = None,
    group_sizes: Optional[List[int]] = None,
    y_scale: Optional[Tensor] = None,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_quant_matmul(Tensor x1, Tensor x2, Tensor scale, *, Tensor? offset=None,
                                 Tensor? pertoken_scale=None, Tensor? bias=None,
                                 ScalarType? output_dtype=None) -> Tensor
    """
    import torch_npu

    if (not is_arch35() and x1.dtype not in [DataType.DT_INT8, DataType.DT_INT32]
        and x2.dtype not in [DataType.DT_INT8, DataType.DT_INT32]):
        raise RuntimeError("In soc versions prior to A5, x1 and x2 only supports int8 or int32.")
    if output_dtype is None:
        output_dtype = 1
    dtype = torch_dtype_value_to_ge_type(output_dtype)

    is_a8w4 = x1.dtype == DataType.DT_FLOAT8_E4M3FN and \
                          (x2_dtype == torch_npu.float4_e2m1fn_x2 or x2.dtype == DataType.DT_FLOAT)
    need_reshape = (x1.dtype == DataType.DT_INT32 and x2.dtype == DataType.DT_INT32) or \
                   (x1_dtype is not None and x2_dtype is not None and \
                    (x1_dtype == torch_npu.float4_e2m1fn_x2 or x1_dtype == torch_npu.float4_e1m2fn_x2)) or \
                   (is_a8w4 and x2.dtype != DataType.DT_FLOAT)
    if need_reshape:
        shape_multiples = 2
        x1_ge_dtype = 0
        x2_ge_dtype = 0
        if x1.dtype == DataType.DT_INT32:
            shape_multiples = 8
            x1_ge_dtype = DataType.DT_INT4
            x2_ge_dtype = DataType.DT_INT4
        else:
            if x1_dtype is not None:
                x1_ge_dtype = torch_dtype_value_to_ge_type(x1_dtype)
            if x2_dtype is not None:
                x2_ge_dtype = torch_dtype_value_to_ge_type(x2_dtype)
        perm = [i for i in range(x2.rank)]
        if(x2.rank < 2):
            raise RuntimeError("Input x2 dimension can't be less than 2, actual x2 dimension is " + str(x2.rank) + ".")
        perm[-1], perm[-2] = perm[-2], perm[-1]
        const_x1 = ge.Const([1] * (x1.rank - 1) + [shape_multiples])
        const_x2 = ge.Const([1] * (x2.rank - 1) + [shape_multiples])
        trans_x2 = x1.symsize[-1] == x2.symsize[-2]

        # A8W4 per-group场景把int64的y_scale转成uint64。 同时，A8W4不需要修改x1的数据类型
        if is_a8w4:
            if pertoken_scale is None and y_scale is not None and y_scale.dtype == DataType.DT_INT64:
                y_scale = ge.Bitcast(y_scale, type=DataType.DT_UINT64)
                trans_x2 = x1.symsize[-1] == (x2.symsize[-2] * 2)
        else:
            shape_x1 = ge.Shape(x1)
            shape_x1 = ge.Mul(shape_x1, const_x1)
            x1 = ge.Bitcast(x1, type=x1_ge_dtype)
            x1 = ge.Reshape(x1, shape_x1)

        if trans_x2:
            x2 = ge.Transpose(x2, perm)
        shape_x2 = ge.Shape(x2)
        shape_x2 = ge.Mul(shape_x2, const_x2)
        x2 = ge.Bitcast(x2, type=x2_ge_dtype)
        x2 = ge.Reshape(x2, shape_x2)
        if trans_x2:
            x2 = ge.Transpose(x2, perm)

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
        if is_a8w4:
            group_size = group_k
    if x1_dtype is not None:
        if x1_dtype != torch_npu.float4_e2m1fn_x2 and x1_dtype != torch_npu.float4_e1m2fn_x2:
            x1 = ge.Bitcast(x1, type=torch_dtype_value_to_ge_type(x1_dtype))
        x1.desc.dtype = torch_dtype_value_to_ge_proto_type(x1_dtype)
    if x2_dtype is not None:
        if x2_dtype != torch_npu.float4_e2m1fn_x2 and x2_dtype != torch_npu.float4_e1m2fn_x2:
            x2 = ge.Bitcast(x2, type=torch_dtype_value_to_ge_type(x2_dtype))
        x2.desc.dtype = torch_dtype_value_to_ge_proto_type(x2_dtype)
    if pertoken_scale_dtype is not None:
        pertoken_scale = ge.Bitcast(pertoken_scale, type=torch_dtype_value_to_ge_type(pertoken_scale_dtype))
        pertoken_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(pertoken_scale_dtype)
    if scale_dtype is not None:
        trans_x2_scale = False
        perm = []
        if is_a8w4 and x2.dtype == DataType.DT_FLOAT:
            trans_x2_scale = x1.symsize[-1] == (x2.symsize[-2] * 8)
            if trans_x2_scale:
                perm = [i for i in range(x2.rank)]
                if(x2.rank < 2):
                    raise RuntimeError("Input x2 dimension can't be less than 2, actual x2 dimension is " + str(x2.rank) + ".")
                perm[-1], perm[-2] = perm[-2], perm[-1]

        if trans_x2_scale:
            scale = ge.Transpose(scale, perm)
        scale = ge.Bitcast(scale, type=torch_dtype_value_to_ge_type(scale_dtype))
        scale.desc.dtype = torch_dtype_value_to_ge_proto_type(scale_dtype)
        if trans_x2_scale:
            scale = ge.Transpose(scale, perm)

    if is_arch35():
        out = ge.QuantBatchMatmulV4(x1,
                                    x2,
                                    bias=bias,
                                    x1_scale=pertoken_scale,
                                    x2_scale=scale,
                                    y_scale=y_scale,
                                    x1_offset=None,
                                    x2_offset=offset,
                                    y_offset=None,
                                    x2_table=None,
                                    dtype=dtype,
                                    transpose_x1=False,
                                    transpose_x2=False,
                                    group_size=group_size)
        out.desc.dtype = torch_dtype_value_to_ge_proto_type(output_dtype)
    else:
        out = ge.QuantBatchMatmulV3(x1,
                                    x2,
                                    scale,
                                    offset=offset,
                                    bias=bias,
                                    pertoken_scale=pertoken_scale,
                                    dtype=dtype,
                                    transpose_x1=False,
                                    transpose_x2=False)
    return out
