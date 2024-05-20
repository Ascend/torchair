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
from torchair._ge_concrete_graph.ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


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
    if bias is None:
        dtype = x[0].dtype
        if dtype == DataType.DT_BF16:
            dtype = DataType.DT_FLOAT
        elif dtype == DataType.DT_UINT8:
            dtype = DataType.DT_INT32
        bias = [ge.Fill([0], ge.Cast(0., dst_type=dtype))]

    if scale is None:
        dtype = DataType.DT_UINT64
        scale = [ge.Fill([0], ge.Cast(0., dst_type=dtype))]

    if offset is None:
        dtype = DataType.DT_FLOAT
        offset = [ge.Fill([0], ge.Cast(0., dst_type=dtype))]

    if antiquant_scale is None:
        x_dtype = x[0].dtype
        w_dtype = weight[0].dtype
        dtype = DataType.DT_FLOAT16
        if (dtype == DataType.DT_FLOAT16 or dtype == DataType.DT_BF16) and w_dtype == DataType.DT_INT8:
            dtype = x_dtype
        antiquant_scale = [ge.Fill([0], ge.Cast(0., dst_type=dtype))]
    
    if antiquant_offset is None:
        x_dtype = x[0].dtype
        w_dtype = weight[0].dtype
        dtype = DataType.DT_FLOAT16
        if (dtype == DataType.DT_FLOAT16 or dtype == DataType.DT_BF16) and w_dtype == DataType.DT_INT8:
            dtype = x_dtype
        antiquant_offset = [ge.Fill([0], ge.Cast(0., dst_type=dtype))]

    y_dtype = -1
    if output_dtype is not None:
        if x[0].dtype == DataType.DT_INT8 and output_dtype == torch.int8:
            y_dtype = 0
        else:
            raise NotImplementedError("In the quant scenario, only int8 is supported for output dtype!")

    if group_list is not None:
        group_list = dtype_promote(group_list, target_dtype=torch.int64)

    return ge.GroupedMatmul(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list,
                            size_of_y=len(meta_outputs), split_item=split_item, dtype=y_dtype, transpose_weight=False)