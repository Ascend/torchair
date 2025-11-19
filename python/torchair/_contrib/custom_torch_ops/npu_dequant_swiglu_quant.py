from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload, Optional
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_type, \
    torch_dtype_value_to_ge_proto_type
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, \
    I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.npu_inference.npu_dequant_swiglu_quant.default)
def conveter_npu_dequant_swiglu_quant_default(
        x: Tensor,
        weight_scale: Tensor = None,
        activation_scale: Tensor = None,
        bias: Tensor = None,
        quant_scale: Tensor = None,
        quant_offset: Tensor = None,
        group_index: Tensor = None,
        activate_left: bool = False,
        quant_mode: int = 0,
        dst_type: int = None,
        round_mode: int = None,
        activate_dim: int = None,
        swiglu_mode: int = 0,
        clamp_limit: float = 7.0,
        glu_alpha: float = 1.702,
        glu_bias: float = 1.0,
        meta_outputs: TensorSpec = None):
    dst_type = dst_type if dst_type is not None else 1
    round_mode = round_mode if round_mode is not None else 0
    activate_dim = activate_dim if activate_dim is not None else -1
    quant_mode_str = 'static'
    if quant_mode == 1:
        quant_mode_str = 'dynamic'

    round_mode_str = "rint"
    if round_mode == 1:
        round_mode_str = "round"
    elif round_mode == 2:
        round_mode_str = "floor"
    elif round_mode == 3:
        round_mode_str = "ceil"
    elif round_mode == 4:
        round_mode_str = "trunc"

    acl_dst_type = torch_dtype_value_to_ge_type(dst_type)
    y, scale = ge.DequantSwigluQuant(x, weight_scale, activation_scale, bias, quant_scale=quant_scale,
                                     quant_offset=quant_offset, group_index=group_index,
                                     activate_left=activate_left, quant_mode=quant_mode_str,
                                     dst_type=acl_dst_type, round_mode=round_mode_str,
                                     activate_dim=activate_dim, swiglu_mode=swiglu_mode, 
                                     clamp_limit=clamp_limit, glu_alpha=glu_alpha, glu_bias=glu_bias)
    y.desc.dtype = torch_dtype_value_to_ge_proto_type(dst_type)
    
    if dst_type == 296 or dst_type == 297:
        dim_num = x.rank
        bit_shape = []
        for _ in range(dim_num - 1):
            bit_shape.append(1)
        bit_shape.append(2)
        tmp_tensor = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
        y_shape_int4 = ge.Shape(y)
        y_shape_uint8 = ge.Div(y_shape_int4, tmp_tensor)
        y_shape_int4_2bit = ge.ConcatV2([y_shape_uint8, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)],
                                        concat_dim=0, N=2)
        y = ge.Bitcast(ge.Reshape(y, y_shape_int4_2bit), type=DataType.DT_UINT8)
        return ge.Reshape(y, y_shape_uint8), scale
    else:
        return y, scale