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


@declare_supported(
    [
        Support(F16(1024, 1024), F16(1024, 1024), hcom="94430305206192", world_size=8, reduce_op="sum", comm_turn=0),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_mm_reduce_scatter_base.default)
def convert_npu_mm_reduce_scatter_base(
    self: Tensor,
    x2: Tensor,
    hcom: str,
    world_size: int,
    reduce_op: str = 'sum',
    bias: Optional[Tensor] = None,
    comm_turn: int = 0,
    meta_outputs: TensorSpec = None
):
    transpose_x1 = False
    transpose_x2 = False
    '''NB: npu::npu_mm_reduce_scatter_base(Tensor self, Tensor x2, str hcom,
       int world_size, *, str reduce_op='sum', Tensor? bias=None, int comm_turn=0) -> Tensor'''
    check_dtype(self, x2, bias=bias)
    return ge.MatmulReduceScatter(self,
                                  x2,
                                  bias=bias,
                                  group=hcom,
                                  reduce_op=reduce_op,
                                  is_trans_a=transpose_x1,
                                  is_trans_b=transpose_x2,
                                  comm_turn=comm_turn)


def check_dtype(x1: Tensor, x2: Tensor, bias: Optional[Tensor]):
    if x1.dtype != x2.dtype:
        raise AssertionError(f"Type of x1:{x1.dtype} and x2:{x2.dtype} must be same.")
    if (x1.dtype != DataType.DT_FLOAT16 and x1.dtype != DataType.DT_BF16):
        raise AssertionError(f"Input supported dtype is fp16/bf16, but got type {x1.dtype}.")