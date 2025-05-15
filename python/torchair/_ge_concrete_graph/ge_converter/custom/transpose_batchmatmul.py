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
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import (_TypedTensor, F32, F16, F64, I32, I16,
                                                               I64, I8, U8, BOOL, Support)


@declare_supported(
    [
        Support(F16(32, 32, 512), F16(32, 512, 128), bias=None, scale=None,
                perm_x1=[1, 0, 2], perm_x2=[0, 1, 2], perm_y=[1, 0, 2]),
        Support(F16(32, 32, 512), F16(32, 512, 128), bias=None, scale=I64(1, 1, 4096),
                perm_x1=[1, 0, 2], perm_x2=[0, 1, 2], perm_y=[1, 0, 2]),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_transpose_batchmatmul.default)
def conveter_npu_npu_transpose_batchmatmul(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    scale: Optional[Tensor] = None,
    perm_x1: Optional[List[int]] = [0, 1, 2],
    perm_x2: Optional[List[int]] = [0, 1, 2],
    perm_y: Optional[List[int]] = [0, 1, 2],
    batch_split_factor: Optional[int] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: npu::npu_transpose_batchmatmul(Tensor input, Tensor weight, *, Tensor? bias=None,
    Tensor? scale=None, int[]? perm_x1=None, int[]? perm_x2=None, int[]? perm_y=None,
    int? batch_split_factor=1) -> Tensor
    """

    return ge.TransposeBatchMatMul(input, weight, bias=bias, scale=scale,
                                   perm_x1=perm_x1, perm_x2=perm_x2, perm_y=perm_y,
                                   enable_hf32=False, batch_split_factor=batch_split_factor)
