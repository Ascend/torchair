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

Y_RANK = 2
TORCH_INT8 = 1
FLOAT4_E2M1 = 296
FLOAT4_E1M2 = 297
TORCH_HIFLOAT8 = 290
TORCH_FLOAT8E4M3 = 23
TORCH_FLOAT8E5M2 = 24


def fill_empty_tensorlist(input_data, desired_dtype):
    if input_data is None:
        return [ge.Fill([0], ge.Cast(0., dst_type=desired_dtype))]
    else:
        return input_data


def convert_tensorlist_to_mxfp4_item(input_data: Tensor, x_dtype, trans=False):
    FP4_IN_INT8 = 2
    x_ge_dtype = 0
    if x_dtype is not None:
        x_ge_dtype = torch_dtype_value_to_ge_type(x_dtype)
    const_x = ge.Const([1] * (input_data.rank - 1) + [FP4_IN_INT8])
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


def convert_tensorlist_to_mxfp4(x: Tensor, weight: List[Tensor], x_dtype, weight_dtype):
    x_new = convert_tensorlist_to_mxfp4_item(x, x_dtype)
    w_list = []
    for w_item in weight:
        new_w = convert_tensorlist_to_mxfp4_item(w_item, weight_dtype)
        w_list.append(new_w)
    return x_new, w_list


def pack_mxfp4_tensor_to_uint8(y: Tensor) -> Tensor:
    dim_num = Y_RANK
    bit_shape = []
    for _ in range(dim_num - 1):
        bit_shape.append(1)
    bit_shape.append(2)
    div_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
    y_shape_int4 = ge.Shape(y)
    y_shape_uint8 = ge.Div(y_shape_int4, div_x2) 
    y_shape_int4_2bit = ge.ConcatV2([y_shape_uint8, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)],
                                        concat_dim=0, N=2)
    y = ge.Bitcast(ge.Reshape(y, y_shape_int4_2bit), type=DataType.DT_UINT8)
    return ge.Reshape(y, y_shape_uint8)


def is_mxfp4(dtype):
    try:
        import torch_npu
    except ImportError as e:
        raise RuntimeError("Couldn't import torch_npu. When the CompilerConfig.mode is reduce-overhead, "
                           "it is necessary to use torch_npu.npu.NPUGraph(), so importing torch_npu is essential. ") from e
    return dtype == torch_npu.float4_e2m1fn_x2 or dtype == torch_npu.float4_e1m2fn_x2


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
    quant_dtype: Optional[int] = 1,
    group_list_type: Optional[int] = 0,
    tuning_config: Optional[List[int]] = None,
    x_dtype: Optional[int] = None,
    weight_dtype: Optional[int] = None,
    weight_scale_dtype: Optional[int] = None,
    x_scale_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    try:
        import torch_npu
    except ImportError as e:
        raise RuntimeError("Couldn't import torch_npu. When the CompilerConfig.mode is reduce-overhead, "
                           "it is necessary to use torch_npu.npu.NPUGraph(), so importing torch_npu is essential. ") from e
    tuning_config = tuning_config or [0]
    weight_assist_matrix = fill_empty_tensorlist(weight_assist_matrix, DataType.DT_FLOAT)
    transpose_weight = False
    x_dtype_weight_dtype_has_value = False
    if x_dtype is not None and weight_dtype is not None:
        x_dtype_weight_dtype_has_value = True
    if x_dtype_weight_dtype_has_value and is_mxfp4(x_dtype) and is_mxfp4(weight_dtype):
        x, weight = convert_tensorlist_to_mxfp4(x, weight, x_dtype, weight_dtype)
    if weight_scale_dtype is not None:
        weight_scale[0] = ge.Bitcast(weight_scale[0], type=torch_dtype_value_to_ge_type(weight_scale_dtype))
        weight_scale[0].desc.dtype = torch_dtype_value_to_ge_proto_type(weight_scale_dtype)
    if x_scale_dtype is not None:
        x_scale = ge.Bitcast(x_scale, type=torch_dtype_value_to_ge_type(x_scale_dtype))
        x_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(x_scale_dtype)
    if x_dtype is not None:
        if x_dtype != torch_npu.float4_e2m1fn_x2 and x_dtype != torch_npu.float4_e1m2fn_x2:
            x = ge.Bitcast(x, type=torch_dtype_value_to_ge_type(x_dtype))
        x.desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype)
    if weight_dtype is not None:
        if weight_dtype != torch_npu.float4_e2m1fn_x2 and weight_dtype != torch_npu.float4_e1m2fn_x2:
            weight[0] = ge.Bitcast(weight[0], type=torch_dtype_value_to_ge_type(weight_dtype))
        weight[0].desc.dtype = torch_dtype_value_to_ge_proto_type(weight_dtype)

    quant_dtype_acl = torch_dtype_value_to_ge_type(quant_dtype)
    dequant_dtype_acl = torch_dtype_value_to_ge_type(dequant_dtype)
    y, y_scale = ge.GroupedMatmulSwigluQuantV2(x=x,
                                        x_scale=x_scale,
                                        group_list=group_list,
                                        weight=weight,
                                        weight_scale=weight_scale,
                                        weight_assist_matrix=weight_assist_matrix,
                                        bias=bias,
                                        smooth_scale=smooth_scale,
                                        dequant_mode=dequant_mode,
                                        dequant_dtype=dequant_dtype_acl,
                                        quant_mode=quant_mode,
                                        quant_dtype=quant_dtype_acl,
                                        transpose_weight=transpose_weight,
                                        group_list_type=group_list_type,
                                        tuning_config=tuning_config
                                       )
    y.desc.dtype = torch_dtype_value_to_ge_proto_type(quant_dtype)
    if quant_dtype != TORCH_INT8: # not torch.int8
        y_scale.desc.dtype = ProtoDataType.DT_FLOAT8_E8M0
    if quant_dtype in (TORCH_INT8, TORCH_HIFLOAT8, TORCH_FLOAT8E4M3, TORCH_FLOAT8E5M2) and quant_mode == 0: 
        y_scale.desc.dtype = ProtoDataType.DT_FLOAT
    if hasattr(torch, "float4_e2m1fn_x2"): # torch2.8.0 support torch.float4_e2m1fn_x2
        if quant_dtype == FLOAT4_E1M2:
            y = pack_mxfp4_tensor_to_uint8(y)
        if quant_dtype == FLOAT4_E2M1:
            y.desc.dtype = torch_dtype_value_to_ge_proto_type(quant_dtype)
    else:
        if quant_dtype in (FLOAT4_E2M1, FLOAT4_E1M2): # torch_npu.float4_e2m1fn_x2 or torch_npu.float4_e1m2fn_x2
            y = pack_mxfp4_tensor_to_uint8(y)
    return y, y_scale