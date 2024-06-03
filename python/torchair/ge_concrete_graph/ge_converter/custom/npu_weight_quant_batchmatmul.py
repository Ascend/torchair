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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import DataType, Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        #支持Abf16W8 bf16输出 | Afp16W8 fp16输出
        # bf16输入时，bias需为fp32 , fp16输入时，bias为fp16
        Support(F16(32, 11264), F16(11264, 1664), F16(1, 1664), F16(1, 1664)),

    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_weight_quant_batchmatmul.default)
def conveter_npu_npu_weight_quant_batchmatmul(
    x: Tensor,
    weight: Tensor,
    antiquant_scale: Tensor,
    antiquant_offset: Optional[Tensor] = None,
    quant_scale: Optional[Tensor] = None,
    quant_offset: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    antiquant_group_size: Optional[int] = 0,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_weight_quant_batchmatmul(Tensor x, Tensor weight, Tensor antiquant_scale,
    Tensor? antiquant_offset=None, Tensor? quant_scale=None, Tensor? quant_offset=None,
    Tensor? bias=None) -> Tensor
    """
    if quant_scale is not None and quant_scale.dtype == DataType.DT_INT64:
        quant_scale = ge.Cast(quant_scale, dst_type=DataType.DT_UINT64)
    if weight is not None and weight.dtype == DataType.DT_INT32:
        perm = [1, 0]
        trans_weight = "Transpose" in weight.tensor
        if trans_weight:
            weight = ge.Transpose(weight, perm)
        shape = ge.Shape(weight)
        const = ge.Const([1, 8])
        weight_shape = ge.Mul(shape, const)
        weight = ge.Bitcast(weight, type=DataType.DT_INT4)
        weight = ge.Reshape(weight, weight_shape)
        if trans_weight:
            weight = ge.Transpose(weight, perm)
    return ge.WeightQuantBatchMatmulV2(x, weight, antiquant_scale, antiquant_offset=antiquant_offset,
                                       quant_scale=quant_scale, quant_offset=quant_offset, bias=bias,
                                       transpose_x=False, transpose_weight=False,
                                       antiquant_group_size=antiquant_group_size)
