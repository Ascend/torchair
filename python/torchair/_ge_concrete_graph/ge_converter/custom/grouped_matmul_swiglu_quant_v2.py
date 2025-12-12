from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload, Optional
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_proto_type, torch_dtype_value_to_ge_type
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, \
    I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType


def fill_empty_tensorlist(input_data, desired_dtype):
    if input_data is None:
        return [ge.Fill([0], ge.Cast(0., dst_type=desired_dtype))]
    else:
        return input_data


@declare_supported([
])
@register_fx_node_ge_converter(torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2.default)
def conveter_npu_grouped_matmul_swiglu_quant_v2(
    x: Tensor,
    weight: List[Tensor],
    weight_scale: List[Tensor],
    x_scale: Tensor,
    group_list: Tensor,
    *,
    smooth_scale: Optional[Tensor] = None,
    weight_assist_matrix: Optional[List[Tensor]] = None,
    bias: Optional[Tensor] = None,
    dequant_mode: Optional[int] = 0,
    dequant_dtype: Optional[int] = 0,
    quant_mode: Optional[int] = 0,
    quant_dtype: Optional[int] = 0,
    group_list_type: Optional[int] = 0,
    tuning_config: Optional[List[int]] = None,
    x_dtype: Optional[int] = None,
    weight_dtype: Optional[int] = None,
    weight_scale_dtype: Optional[int] = None,
    x_scale_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):

    tuning_config = tuning_config or [0]
    weight_assist_matrix = fill_empty_tensorlist(weight_assist_matrix, DataType.DT_FLOAT)
    transpose_weight = False
    if weight_scale_dtype is not None:
        weight_scale[0] = ge.Bitcast(weight_scale[0], type=torch_dtype_value_to_ge_type(weight_scale_dtype))
        weight_scale[0].desc.dtype = torch_dtype_value_to_ge_proto_type(weight_scale_dtype)
    if x_scale_dtype is not None:
        x_scale = ge.Bitcast(x_scale, type=torch_dtype_value_to_ge_type(x_scale_dtype))
        x_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(x_scale_dtype)
    acl_dst_type = torch_dtype_value_to_ge_type(quant_dtype)
    y, y_scale = ge.GroupedMatmulSwigluQuantV2(x=x,
                                        x_scale=x_scale,
                                        group_list=group_list,
                                        weight=weight,
                                        weight_scale=weight_scale,
                                        weight_assist_matrix=weight_assist_matrix,
                                        bias=bias,
                                        smooth_scale=smooth_scale,
                                        dequant_mode=dequant_mode,
                                        dequant_dtype=dequant_dtype,
                                        quant_mode=quant_mode,
                                        quant_dtype=acl_dst_type,
                                        transpose_weight=transpose_weight,
                                        group_list_type=group_list_type,
                                        tuning_config=tuning_config
                                       )
    y.desc.dtype = torch_dtype_value_to_ge_proto_type(quant_dtype)
    y_scale.desc.dtype = ProtoDataType.DT_FLOAT8_E8M0
    return y, y_scale