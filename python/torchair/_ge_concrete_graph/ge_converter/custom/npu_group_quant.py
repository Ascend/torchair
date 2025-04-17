from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter, \
    torch_type_to_ge_type
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, \
    I64, I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2, 2), F32(2, 2), I32(2), F32(2), dst_dtype=torch.qint8),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_group_quant.default)
def conveter_npu_group_quant_default(
    x: Tensor,
    scale: Tensor,
    group_index: Tensor,
    offset: Optional[Tensor] = None,
    dst_dtype: int = torch.uint8,
    meta_outputs: TensorSpec = None
):
    """
    NB: aten::npu_group_quant(Tensor x, Tensor scale, Tensor group_index, *, Tensor? offset,
                              ScalarType? dst_dtype) -> Tensor
    """
    if dst_dtype == torch.qint8:
        dst_dtype = torch.int8

    y = ge.GroupQuant(x, scale, group_index, offset=offset, dst_type=torch_type_to_ge_type(dst_dtype))
    if dst_dtype == torch.quint4x2:
        dim_num = x.rank
        bit_shape = []
        for _ in range(dim_num - 1):
            bit_shape.append(1)
        bit_shape.append(8)
        # y int4 shape is (..., 8n), y int32 shape is (..., n), y bitcast shape is (..., n, 8)
        y_shape_int32 = ge.Div(ge.Shape(y), ge.Const(bit_shape, dtype=DataType.DT_INT32))
        y_shape_int4_bitcast = ge.ConcatV2([y_shape_int32, ge.Const([8], dtype=DataType.DT_INT32)],
                                        concat_dim=0, N=2)
        y = ge.Bitcast(ge.Reshape(y, y_shape_int4_bitcast), type=DataType.DT_INT32)
        return ge.Reshape(y, y_shape_int32)
    else:
        return y
