from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_type, \
torch_dtype_value_to_ge_proto_type, _ge_dtype_to_ge_proto_dtype

DTYPE_SUPPORT_LIST_QUANT = {DataType.DT_FLOAT8_E4M3FN, DataType.DT_FLOAT8_E5M2, DataType.DT_HIFLOAT8,
                            DataType.DT_UINT8, DataType.DT_INT8}
DTYPE_SUPPORT_LIST_WEIGHT_QUANT_X2 = {DataType.DT_FLOAT8_E4M3FN, DataType.DT_HIFLOAT8, DataType.DT_INT8}
# A16W16/A16W8/A16w4: x1/bias only support bf16/fp16, and 2 types must be same
DTYPE_SUPPORT_BIAS = {DataType.DT_BF16, DataType.DT_FLOAT16}
DTYPE_SUPPORT_X1 = {DataType.DT_BF16, DataType.DT_FLOAT16}


@declare_supported(
    [
        Support(F16(1024, 1024), F16(1024, 1024), "group", reduce_op="sum", bias=F16(1024), comm_turn=0),
        Support(I8(1024, 1024), I8(1024, 1024), "group", reduce_op="sum", bias=I32(1024), dequant_scale=I64(1),
                comm_turn=0),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_mm_all_reduce_base.default)
def convert_npu_mm_all_reduce_base(
    x1: Tensor,
    x2: Tensor,
    hcom: str,
    *,
    reduce_op: str = 'sum',
    bias: Optional[Tensor] = None,
    antiquant_scale: Optional[Tensor] = None,
    antiquant_offset: Optional[Tensor] = None,
    x3: Optional[Tensor] = None,
    dequant_scale: Optional[Tensor] = None,
    pertoken_scale: Optional[Tensor] = None,
    comm_quant_scale_1: Optional[Tensor] = None,
    comm_quant_scale_2: Optional[Tensor] = None,
    antiquant_group_size: int = 0,
    comm_turn: int = 0,
    group_sizes: Optional[List[int]] = None,
    y_dtype: int = None, 
    x1_dtype: int = None, 
    x2_dtype: int = None, 
    dequant_scale_dtype: int = None, 
    pertoken_scale_dtype: int = None, 
    comm_quant_mode: int = 0,
    meta_outputs: TensorSpec = None
):
    # transpose_x1 is set to False by default
    transpose_x1 = False
    transpose_x2 = False
    '''NB: npu::npu_mm_all_reduce_base(Tensor x1, Tensor x2, str hcom, *, str reduce_op='sum', Tensor? bias=None,
                                       Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? x3=None,
                                       Tensor? dequant_scale=None, Tensor? pertoken_scale=None,
                                       Tensor? comm_quant_scale_1=None, Tensor? comm_quant_scale_2=None,
                                       int? antiquant_group_size=0, int? comm_turn=0, 
                                       int[]? group_sizes=None, int? y_dtype=None, int? x1_dtype=None, int? x2_dtype=None, 
                                       int? dequant_scale_dtype=None, int? pertoken_scale_dtype=None, int? comm_quant_mode=0) -> Tensor'''
    import torch_npu

    if dequant_scale is not None:
        check_dtype_full_quant(x1, x2, bias=bias, x3=x3, dequant_scale=dequant_scale,
                               x1_dtype=x1_dtype, x2_dtype=x2_dtype)
    if antiquant_scale is not None:
        check_dtype_weight_quant(x1, x2, bias=bias, x3=x3, x2_dtype=x2_dtype)
    if dequant_scale is None and antiquant_scale is None:
        check_dtype_non_quant(x1, x2, bias, x3)

    if y_dtype is None:
        if dequant_scale is not None:
            ge_y_dtype = DataType.DT_BF16 if dequant_scale.dtype == DataType.DT_BF16 else DataType.DT_FLOAT16
        else:
            ge_y_dtype = DataType.DT_UNDEFINED
    else:
        ge_y_dtype = torch_dtype_value_to_ge_type(y_dtype)
    output_dtype = x1.dtype if ge_y_dtype == DataType.DT_UNDEFINED else ge_y_dtype

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

    shape_multiples = 2
    if (x1.rank < 2):
        raise RuntimeError("Input x1 dimension can't be less than 2, actual x1 dimension is " + str(x1.rank) + ".")
    if (x2.rank < 2):
        raise RuntimeError("Input x2 dimension can't be less than 2, actual x2 dimension is " + str(x2.rank) + ".")
    trans_x2 = x1.symsize[-1] == x2.symsize[-2]
    if x1_dtype is not None:
        x1_ge_dtype = torch_dtype_value_to_ge_type(x1_dtype)
        if x1_dtype == torch_npu.float4_e2m1fn_x2:
            const_x1 = ge.Const([1] * (x1.rank - 1) + [shape_multiples])
            shape_x1 = ge.Shape(x1)
            shape_x1 = ge.Mul(shape_x1, const_x1)
            x1 = ge.Bitcast(x1, type=x1_ge_dtype)
            x1 = ge.Reshape(x1, shape_x1)
        else:
            x1 = ge.Bitcast(x1, type=x1_ge_dtype)
        x1.desc.dtype = torch_dtype_value_to_ge_proto_type(x1_dtype)
    if x2_dtype is not None:
        x2_ge_dtype = torch_dtype_value_to_ge_type(x2_dtype)
        if x2_dtype == torch_npu.float4_e2m1fn_x2:
            perm = [i for i in range(x2.rank)]
            perm[-2], perm[-1] = perm[-1], perm[-2]
            const_x2 = ge.Const([1] * (x2.rank - 1) + [shape_multiples])
            if trans_x2:
                x2 = ge.Transpose(x2, perm)
            shape_x2 = ge.Shape(x2)
            shape_x2 = ge.Mul(shape_x2, const_x2)
            x2 = ge.Bitcast(x2, type=x2_ge_dtype)
            x2 = ge.Reshape(x2, shape_x2)
            if trans_x2:
                x2 = ge.Transpose(x2, perm)
        else:
            x2 = ge.Bitcast(x2, type=x2_ge_dtype)
        x2.desc.dtype = torch_dtype_value_to_ge_proto_type(x2_dtype)
    if dequant_scale_dtype is not None:
        dequant_scale = ge.Bitcast(dequant_scale, type=torch_dtype_value_to_ge_type(dequant_scale_dtype))
        dequant_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(dequant_scale_dtype)
    if pertoken_scale_dtype is not None:
        pertoken_scale = ge.Bitcast(pertoken_scale, type=torch_dtype_value_to_ge_type(pertoken_scale_dtype))
        pertoken_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(pertoken_scale_dtype)

    out = ge.MatmulAllReduce(x1,
                             x2,
                             bias=bias,
                             x3=x3,
                             antiquant_scale=antiquant_scale,
                             antiquant_offset=antiquant_offset,
                             dequant_scale=dequant_scale,
                             pertoken_scale=pertoken_scale,
                             comm_quant_scale_1=comm_quant_scale_1,
                             comm_quant_scale_2=comm_quant_scale_2,
                             group=hcom,
                             reduce_op=reduce_op,
                             is_trans_a=transpose_x1,
                             is_trans_b=transpose_x2,
                             comm_turn=comm_turn,
                             antiquant_group_size=antiquant_group_size,
                             group_size=group_size,
                             y_dtype=ge_y_dtype, 
                             comm_quant_mode=comm_quant_mode)
    out.desc.dtype = _ge_dtype_to_ge_proto_dtype(output_dtype)
    return out


def check_dtype_non_quant(x1: Tensor, x2: Tensor, bias: Optional[Tensor], x3: Optional[Tensor]):
    if x1.dtype not in DTYPE_SUPPORT_X1:
        raise AssertionError(
            f"Non-quant scene: x1 dtype {ge_type_to_torch_type(x1.dtype)} not supported, only support "
            f"{[ge_type_to_torch_type(d) for d in DTYPE_SUPPORT_X1]}.")
    if x2.dtype != x1.dtype:
        raise AssertionError(
            f"Non-quant scene: x1 dtype {ge_type_to_torch_type(x1.dtype)} != x2 dtype "
            f"{ge_type_to_torch_type(x2.dtype)}, must be same.")
    if bias is not None and bias.dtype != x1.dtype:
        raise AssertionError(
            f"Non-quant scene: x1 dtype {ge_type_to_torch_type(x1.dtype)} != bias dtype "
            f"{ge_type_to_torch_type(bias.dtype)}, must be same.")
    if x3 is not None and x3.dtype != x1.dtype:
        raise AssertionError(
            f"Non-quant scene: x1 dtype {ge_type_to_torch_type(x1.dtype)} != x3 dtype "
            f"{ge_type_to_torch_type(x3.dtype)}, must be same.")


def check_dtype_weight_quant(x1: Tensor, x2: Tensor, bias: Optional[Tensor], x3: Optional[Tensor], x2_dtype: int):
    if x1.dtype not in DTYPE_SUPPORT_X1:
        raise AssertionError(
            f"Weight quant scene: x1 dtype {ge_type_to_torch_type(x1.dtype)} not supported, "
            f"only support {[ge_type_to_torch_type(d) for d in DTYPE_SUPPORT_X1]}.")
    if bias is not None and bias.dtype != x1.dtype:
        raise AssertionError(
            f"Weight quant scene: x1 dtype {ge_type_to_torch_type(x1.dtype)} != bias dtype "
            f"{ge_type_to_torch_type(bias.dtype)}, must be same.")
    if x3 is not None and x3.dtype != x1.dtype:
        raise AssertionError(
            f"Weight quant scene: x1 dtype {ge_type_to_torch_type(x1.dtype)} != x3 dtype "
            f"{ge_type_to_torch_type(x3.dtype)}, must be same.")
    if x2.dtype not in DTYPE_SUPPORT_LIST_WEIGHT_QUANT_X2:
        raise AssertionError(
            f"Weight quant scene: x2 dtype {ge_type_to_torch_type(x2.dtype)} not supported, "
            f"support list: {[ge_type_to_torch_type(d) for d in DTYPE_SUPPORT_LIST_WEIGHT_QUANT_X2]}.")


def check_dtype_full_quant(x1: Tensor, x2: Tensor, bias: Optional[Tensor], x3: Optional[Tensor],
                           dequant_scale: Optional[Tensor], x1_dtype: int, x2_dtype: int):
    if x1.dtype not in DTYPE_SUPPORT_LIST_QUANT:
        raise AssertionError(
            f"Full quant scene: x1 dtype {ge_type_to_torch_type(x1.dtype)} not supported, "
            f"support list: {[ge_type_to_torch_type(d) for d in DTYPE_SUPPORT_LIST_QUANT]}.")
    if x2.dtype not in DTYPE_SUPPORT_LIST_QUANT:
        raise AssertionError(
            f"Full quant scene: x2 dtype {ge_type_to_torch_type(x2.dtype)} not supported, "
            f"support list: {[ge_type_to_torch_type(d) for d in DTYPE_SUPPORT_LIST_QUANT]}.")

    if x1.dtype == DataType.DT_UINT8 and x2.dtype == DataType.DT_UINT8:
        x1_gedtype = torch_dtype_value_to_ge_type(x1_dtype) if x1_dtype is not None else None
        x2_gedtype = torch_dtype_value_to_ge_type(x2_dtype) if x2_dtype is not None else None
        if not ((x1_gedtype == DataType.DT_FLOAT4_E2M1 and x2_gedtype == DataType.DT_FLOAT4_E2M1) or \
                (x1_gedtype == DataType.DT_HIFLOAT8 and x2_gedtype == DataType.DT_HIFLOAT8)):
            raise AssertionError(
                f"Full quant scene: x1/x2 dtype are both uint8, x1_dtype/x2_dtype must be "
                f"float4_e2m1fn_x2/hifloat8, actual x1_dtype={x1_dtype}, x2_dtype={x2_dtype}.")
