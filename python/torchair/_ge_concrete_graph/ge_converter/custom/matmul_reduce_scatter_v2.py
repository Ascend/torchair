from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_proto_type, \
    torch_dtype_value_to_ge_type
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support

X1_X2_SAME_TYPES = {
    DataType.DT_FLOAT16,
    DataType.DT_BF16,
    DataType.DT_HIFLOAT8
}

FP8_TYPES = {
    DataType.DT_HIFLOAT8,
    DataType.DT_FLOAT8_E4M3FN,
    DataType.DT_FLOAT8_E5M2
}


@declare_supported([
        Support(F16(1024, 1024),
                F16(1024, 1024),
                hcom="94430305206192",
                world_size=8,
                reduce_op="sum",
                bias=None,
                x1_scale=None,
                x2_scale=None,
                quant_scale=None,
                block_size=0,
                comm_turn=0,
                group_sizes=None,
                amax_output=False,
                y_dtype=None,
                x1_dtype=None,
                x2_dtype=None,
                x1_scale_dtype=None,
                x2_scale_dtype=None)
])

@register_fx_node_ge_converter(torch.ops.npu.npu_quant_mm_reduce_scatter.default)
def convert_npu_quant_mm_reduce_scatter(
    self: Tensor,
    x2: Tensor,
    hcom: str,
    world_size: int,
    reduce_op: str = 'sum',
    bias: Optional[Tensor] = None,
    x1_scale: Optional[Tensor] = None,
    x2_scale: Optional[Tensor] = None,
    quant_scale: Optional[Tensor] = None,
    block_size: int = 0,
    comm_turn: int = 0,
    group_sizes: Optional[List[int]] = None,
    amax_output: bool = False,
    y_dtype: int = None,
    x1_dtype: int = None,
    x2_dtype: int = None,
    x1_scale_dtype: int = None,
    x2_scale_dtype: int = None,
    meta_outputs: TensorSpec = None
):
    transpose_x1 = False
    transpose_x2 = False
    '''NB: npu::npu_quant_mm_reduce_scatter(Tensor self, Tensor x2, str hcom,
                                            int world_size, *, str reduce_op='sum', Tensor? bias=None,
                                            Tensor? x1_scale=None, Tensor? x2_scale=None,
                                            Tensor? quant_scale=None, int block_size=0, int comm_turn=0,
                                            int[]? group_sizes=None, bool amax_output=False, int? y_dtype=None,
                                            int? x1_dtype=None, int? x2_dtype=None, int? x1_scale_dtype=None,
                                            int? x2_scale_dtype=None) -> (Tensor, Tensor)'''
    check_dtype(self, x2, y_dtype=y_dtype)
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
    output_dtype = self.dtype if y_dtype is None else torch_dtype_value_to_ge_type(y_dtype)
    if y_dtype is None:
        y_dtype = 1
    if x1_dtype is not None:
        self = ge.Bitcast(self, type=torch_dtype_value_to_ge_type(x1_dtype))
        self.desc.dtype = torch_dtype_value_to_ge_proto_type(x1_dtype)
    if x2_dtype is not None:
        x2 = ge.Bitcast(x2, type=torch_dtype_value_to_ge_type(x2_dtype))
        x2.desc.dtype = torch_dtype_value_to_ge_proto_type(x2_dtype)
    if x1_scale_dtype is not None:
        x1_scale = ge.Bitcast(x1_scale, type=torch_dtype_value_to_ge_type(x1_scale_dtype))
        x1_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(x1_scale_dtype)
    if x2_scale_dtype is not None:
        x2_scale = ge.Bitcast(x2_scale, type=torch_dtype_value_to_ge_type(x2_scale_dtype))
        x2_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(x2_scale_dtype)
    (out, amax_out) = ge.MatmulReduceScatterV2(self,
                                               x2,
                                               bias=bias,
                                               x1_scale=x1_scale,
                                               x2_scale=x2_scale,
                                               quant_scale=quant_scale,
                                               group=hcom,
                                               reduce_op=reduce_op,
                                               is_trans_a=transpose_x1,
                                               is_trans_b=transpose_x2,
                                               comm_turn=comm_turn,
                                               rank_size=world_size,
                                               block_size=block_size,
                                               group_size=group_size,
                                               is_amax_out=amax_output,
                                               y_dtype=output_dtype)
    # 对于非原生的torch数据类型需要做类型标注
    out.desc.dtype = torch_dtype_value_to_ge_proto_type(y_dtype)
    return (out, amax_out)


def check_dtype(x1: Tensor, x2: Tensor, y_dtype: int = None):
    if (x1.dtype != x2.dtype) and x1.dtype in X1_X2_SAME_TYPES:
        raise AssertionError(f"Type of x1:{x1.dtype} and x2:{x2.dtype} must be same.")
    if x1.dtype in FP8_TYPES and y_dtype is None:
        raise AssertionError(f"When type of x1 is {x1.dtype} should input y_dtype.")