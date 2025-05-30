from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
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


def convert_tensorlist_to_int4(input_data: List[Tensor]):
    input_dtype = input_data[0].dtype
    w_list = []
    if input_dtype == DataType.DT_INT32:
        for w_item in input_data:
            const_w = ge.Const([1] * (w_item.rank - 1) + [8])
            shape_w = ge.Shape(w_item)
            shape_w = ge.Mul(shape_w, const_w)
            new_w = ge.Bitcast(w_item, type=DataType.DT_INT4)
            new_w = ge.Reshape(new_w, shape_w)
            w_list.append(new_w)
    else:
        w_list = input_data
    return w_list


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


def convert_ydtype(output_dtype: Optional[int] = None):
    y_dtype = -1
    if output_dtype == torch.float16:
        y_dtype = 0
    elif output_dtype == torch.bfloat16:
        y_dtype = 1
    elif output_dtype == torch.int32:
        y_dtype = 2
    elif output_dtype == torch.int8:
        raise ValueError("output_dtype not support int8 yet for graph mode")
    else:
        raise ValueError(f"output_dtype should be float16, bfloat16 or int32, "
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
    meta_outputs: TensorSpec = None,
):
    """NB: npu::npu_grouped_matmul(x, weight, *, bias=None, scale=None, offset=None, antiquant_scale=None, 
            antiquant_offset=None, per_token_scale=None, group_list=None,
            activation_input=None, activation_quant_offset=None, activation_quant_offset=None, 
            split_item=0, group_type=-1, group_list_type=0, act_type=0,
            tuning_config=[0], output_dtype=None) -> Tensor[]
    """
    tuning_config = tuning_config or [0]
    x_dtype = x[0].dtype

    if x_dtype == DataType.DT_BF16:
        bias = fill_empty_tensorlist(bias, DataType.DT_FLOAT)
    elif x_dtype == DataType.DT_INT8:
        bias = fill_empty_tensorlist(bias, DataType.DT_INT32)
    else:
        bias = fill_empty_tensorlist(bias, x_dtype)

    scale = fill_empty_tensorlist(scale, DataType.DT_UINT64)
    scale = convert_scale_tensorlist(scale)
    offset = fill_empty_tensorlist(offset, DataType.DT_FLOAT)

    w_dtype = weight[0].dtype

    if (x_dtype == DataType.DT_FLOAT16 or x_dtype == DataType.DT_BF16) and w_dtype == DataType.DT_INT8:
        antiquant_scale = fill_empty_tensorlist(antiquant_scale, x_dtype)
        antiquant_offset = fill_empty_tensorlist(antiquant_offset, x_dtype)
    else:
        antiquant_scale = fill_empty_tensorlist(antiquant_scale, DataType.DT_FLOAT16)
        antiquant_offset = fill_empty_tensorlist(antiquant_offset, DataType.DT_FLOAT16)

    y_dtype = -1
    if output_dtype is not None and x[0].dtype == DataType.DT_INT8:
        y_dtype = convert_ydtype(output_dtype)

    per_token_scale = per_token_scale[0] if per_token_scale is not None and len(per_token_scale) else None

    if group_list is not None and not isinstance(group_list, Tensor):
        group_list = dtype_promote(group_list, target_dtype=torch.int64)

    w_list = convert_tensorlist_to_int4(weight)

    return ge.GroupedMatmul(x, w_list, bias, scale, offset, antiquant_scale, antiquant_offset, group_list,
                            per_token_scale, split_item=split_item, dtype=y_dtype, transpose_weight=False,
                            transpose_x=False, group_type=group_type, group_list_type=group_list_type,
                            act_type=act_type, tuning_config=tuning_config)


gmm_reg = register_fx_node_ge_converter(torch.ops.npu.npu_grouped_matmul.default)(conveter_npu_npu_grouped_matmul)
gmm_List_reg = register_fx_node_ge_converter(torch.ops.npu.npu_grouped_matmul.List)(conveter_npu_npu_grouped_matmul)

declare_supported([
    Support([F16(8192, 320)], [F16(320, 2560)], bias=[F16(2560)], split_item=0, tuning_config=[0]),
    Support([F16(8192, 320), F16(8192, 320), F16(8192, 320)],
            [F16(320, 2560), F16(320, 2560), F16(320, 2560)],
            bias=[F16(2560), F16(2560), F16(2560)], split_item=0, tuning_config=[0]),
])(gmm_reg)

declare_supported([
    Support([F16(8192, 320)], [F16(320, 2560)], bias=[F16(2560)], split_item=0),
    Support([F16(8192, 320), F16(8192, 320), F16(8192, 320)],
            [F16(320, 2560), F16(320, 2560), F16(320, 2560)],
            bias=[F16(2560), F16(2560), F16(2560)], split_item=0),
])(gmm_List_reg)