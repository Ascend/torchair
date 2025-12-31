from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_type, \
torch_dtype_value_to_ge_proto_type, _ge_dtype_to_ge_proto_dtype

DTYPE_SUPPORT_LIST_QUANT_FP4 = {DataType.DT_FLOAT4_E1M2, DataType.DT_FLOAT4_E2M1}
DTYPE_SUPPORT_LIST_QUANT_FP8 = {DataType.DT_FLOAT8_E4M3FN, DataType.DT_FLOAT8_E5M2, DataType.DT_HIFLOAT8}


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
    
    check_dtype(x1, x2, bias=bias, x3=x3, antiquant_scale=antiquant_scale,
                antiquant_offset=antiquant_offset, dequant_scale=dequant_scale, y_dtype=y_dtype)

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
        if x1_dtype == torch_npu.float4_e2m1fn_x2 or x1_dtype == torch_npu.float4_e1m2fn_x2:
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
        if x2_dtype == torch_npu.float4_e2m1fn_x2 or x2_dtype == torch_npu.float4_e1m2fn_x2:
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


def check_dtype(x1: Tensor, x2: Tensor, bias: Optional[Tensor], x3: Optional[Tensor],
                antiquant_scale: Optional[Tensor], antiquant_offset: Optional[Tensor],
                dequant_scale: Optional[Tensor], y_dtype: int = None):
    if (x1.dtype in DTYPE_SUPPORT_LIST_QUANT_FP8 or x1.dtype in DTYPE_SUPPORT_LIST_QUANT_FP4) and y_dtype is None:
        raise AssertionError(f"When type of x1 is:{x1.dtype} , should input y_dtype.")
    if (x1.dtype == DataType.DT_FLOAT16 or x1.dtype == DataType.DT_BF16) and \
        (x2.dtype == DataType.DT_FLOAT16 or x2.dtype == DataType.DT_BF16):
        if x2.dtype != x1.dtype:
            raise AssertionError(f"type of x1:{x1.dtype} and x2:{x2.dtype} must be same.")
        if bias is not None and bias.dtype != x1.dtype:
            raise AssertionError(f"type of x1:{x1.dtype} and bias:{bias.dtype} must be same.")
        if x3 is not None and x3.dtype != x1.dtype:
            raise AssertionError(f"type of x1:{x1.dtype} and x3:{x3.dtype} must be same.")