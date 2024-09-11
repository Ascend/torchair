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
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge._ge_graph import DataType


@declare_supported([
    Support(I8(8192, 320),
            I8(320, 2560),
            F32(1),
            offset=None,
            pertoken_scale=None,
            bias=None,
            output_dtype=torch.int8,
            transpose_x1=False,
            transpose_x2=False)
])

@register_fx_node_ge_converter(torch.ops.npu.npu_quant_matmul.default)
def conveter_npu_npu_quant_matmul(
    x1: Tensor,
    x2: Tensor,
    scale: Tensor,
    *,
    offset: Optional[Tensor] = None,
    pertoken_scale: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    output_dtype: torch.dtype = None,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_quant_matmul(Tensor x1, Tensor x2, Tensor scale, *, Tensor? offset=None,
                                 Tensor? pertoken_scale=None, Tensor? bias=None,
                                 ScalarType? output_dtype=None) -> Tensor
    """
    dtype = DataType.DT_INT8
    if output_dtype is None or output_dtype == torch.int8:
        dtype = DataType.DT_INT8
    elif output_dtype == torch.float16:
        dtype = DataType.DT_FLOAT16
    elif output_dtype == torch.bfloat16:
        dtype = DataType.DT_BF16
    else:
        raise RuntimeError("Not supported output dtype is " + str(output_dtype))

    if x1.dtype == DataType.DT_INT32 and x2.dtype == DataType.DT_INT32:
        perm = [1, 0]
        # 1个INT32数据存储8个有效的INT4数据，数据类型转换后内轴维度放大8倍
        const_x1 = ge.Const([1] * (x1.rank - 1) + [8])
        const_x2 = ge.Const([1] * (x2.rank - 1) + [8])
        trans_x2 = x1.symsize[-1] == x2.symsize[-2]

        shape_x1 = ge.Shape(x1)
        shape_x1 = ge.Mul(shape_x1, const_x1)
        x1 = ge.Bitcast(x1, type=DataType.DT_INT4)
        x1 = ge.Reshape(x1, shape_x1)

        if trans_x2:
            x2 = ge.Transpose(x2, perm)
        shape_x2 = ge.Shape(x2)
        shape_x2 = ge.Mul(shape_x2, const_x2)
        x2 = ge.Bitcast(x2, type=DataType.DT_INT4)
        x2 = ge.Reshape(x2, shape_x2)
        if trans_x2:
            x2 = ge.Transpose(x2, perm)

    return ge.QuantBatchMatmulV3(x1,
                                 x2,
                                 scale,
                                 offset=offset,
                                 bias=bias,
                                 pertoken_scale=pertoken_scale,
                                 dtype=dtype,
                                 transpose_x1=False,
                                 transpose_x2=False)
