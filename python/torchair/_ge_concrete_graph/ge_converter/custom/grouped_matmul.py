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
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


def fill(dims: Tensor, value: Tensor):
    """REG_OP(fill)\n
    .INPUT(dims, TensorType::IndexNumberType()\n
    .INPUT(value, TensorType({{DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, 
           DT_UINT16, DT_UINT32, DT_UINT64, DT_DOUBLE, DT_COMPLEX64, DT_BOOL, DT_STRING}))\n
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, 
            DT_UINT16, DT_UINT32, DT_UINT64, DT_DOUBLE, DT_COMPLEX64, DT_BOOL, DT_STRING}))\n
    """

    y = torchair.ge.custom_op("fill",
        inputs={ 
            "dims":dims,
            "value":value
        },
        outputs=["y"]
    )
    return y


def fill_empty_tensorlist(input_data, desired_dtype):
    if input_data is None:
        return [fill(ge.Const(0), ge.Const(0., dst_type=desired_dtype))]
    else:
        return input_data


@declare_supported([
    Support([F16(8192, 320)], [F16(320, 2560)], [F16(2560)], None, None, None, None, None, split_item=0),
    Support([F16(8192, 320), F16(8192, 320), F16(8192, 320)],
            [F16(320, 2560), F16(320, 2560), F16(320, 2560)],
            [F16(2560), F16(2560), F16(2560)],
            None, None, None, None, None, split_item=0),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_grouped_matmul.default)
def conveter_npu_npu_grouped_matmul(
    x: List[Tensor],
    weight: List[Tensor],
    *,
    bias: Optional[List[Tensor]] = None,
    scale: Optional[List[Tensor]] = None,
    offset: Optional[List[Tensor]] = None,
    antiquant_scale: Optional[List[Tensor]] = None,
    antiquant_offset: Optional[List[Tensor]] = None,
    group_list: Optional[List[int]] = None,
    activation_input: Optional[List[Tensor]] = None,
    activation_quant_scale: Optional[List[Tensor]] = None,
    activation_quant_offset: Optional[List[Tensor]] = None,
    split_item: Optional[int] = 0,
    group_type: Optional[int] = -1,
    group_list_type: Optional[int] = 0,
    act_type: Optional[int] = 0,
    output_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: npu::npu_grouped_matmul(x, weight, *, bias=None, scale=None, offset=None, antiquant_scale=None, 
            antiquant_offset=None, per_token_scale=None, group_list=None,
            activation_input=None, activation_quant_offset=None, activation_quant_offset=None, 
            split_item=0, group_type=-1, group_list_type=0, act_type=0, output_dtype=None) -> Tensor[]
    """
    x_dtype = x[0].dtype

    if x_dtype == DataType.DT_BF16:
        bias = fill_empty_tensorlist(bias, DataType.DT_FLOAT)
    elif x_dtype == DataType.DT_UINT8:
        bias = fill_empty_tensorlist(bias, DataType.DT_INT32)
    else:
        bias = fill_empty_tensorlist(bias, x_dtype)

    scale = fill_empty_tensorlist(scale, DataType.DT_UINT64)
    offset = fill_empty_tensorlist(offset, DataType.DT_FLOAT)

    w_dtype = weight[0].dtype

    if x_dtype == DataType.DT_BF16:
        antiquant_scale = fill_empty_tensorlist(antiquant_scale, DataType.DT_BF16)
        antiquant_offset = fill_empty_tensorlist(antiquant_offset, DataType.DT_BF16)
    else:
        antiquant_scale = fill_empty_tensorlist(antiquant_scale, DataType.DT_FLOAT16)
        antiquant_offset = fill_empty_tensorlist(antiquant_offset, DataType.DT_FLOAT16)

    y_dtype = -1
    if output_dtype is None or output_dtype == torch.float16:
        y_dtype = 0
    elif output_dtype == torch.bfloat16:
        y_dtype = 1
    elif output_dtype == torch.int8:
        raise ValueError("output_dtype not support int8 yet for graph mode")
    else:
        raise ValueError(f"output_dtype should be int8, float16 or bfloat16, "
                             f"otherwise it should be None, but got {output_dtype}")

    if group_list is not None:
        group_list = dtype_promote(group_list, target_dtype=torch.int64)

    return ge.GroupedMatmul(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list,
                            per_token_scale, split_item=split_item, dtype=y_dtype, transpose_weight=False, 
                            transpose_x=False, group_type=group_type)