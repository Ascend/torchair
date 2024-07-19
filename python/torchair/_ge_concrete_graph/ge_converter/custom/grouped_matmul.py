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


def fill_empty_tensorlist(input_data, desired_dtype):
    if input_data is None:
        return [ge.Fill([0], ge.Cast(0., dst_type=desired_dtype))]
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
    split_item: Optional[int] = 0,
    output_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: npu::npu_grouped_matmul(Tensor[] x, Tensor[] weight, *, Tensor[]? bias, Tensor[]? scale, Tensor[]? offset,
    Tensor[]? antiquant_scale, Tensor[]? antiquant_offset, int[]? group_list=None, int? split_item=0,
    ScalarType? output_dtype=None) -> Tensor[]
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

    if (x_dtype == DataType.DT_FLOAT16 or x_dtype == DataType.DT_BF16) and w_dtype == DataType.DT_INT8:
        antiquant_scale = fill_empty_tensorlist(antiquant_scale, x_dtype)
        antiquant_offset = fill_empty_tensorlist(antiquant_offset, x_dtype)
    else:
        antiquant_scale = fill_empty_tensorlist(antiquant_scale, DataType.DT_FLOAT16)
        antiquant_offset = fill_empty_tensorlist(antiquant_offset, DataType.DT_FLOAT16)

    y_dtype = -1
    if output_dtype is not None:
        if x[0].dtype == DataType.DT_INT8 and output_dtype == torch.int8:
            y_dtype = 0
        else:
            raise NotImplementedError("In the quant scenario, only int8 is supported for output dtype!")

    if group_list is not None:
        group_list = dtype_promote(group_list, target_dtype=torch.int64)

    group_type = -1 # -1 means input tensors no not need to be split
    if len(x) != len(weight) or split_item == 2 or split_item == 3: # When split_item = 2 or 3, output is single tensor
        group_type = 0

    return ge.GroupedMatmul(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list,
                            split_item=split_item, dtype=y_dtype, transpose_weight=False, transpose_x=False,
                            group_type=group_type)

