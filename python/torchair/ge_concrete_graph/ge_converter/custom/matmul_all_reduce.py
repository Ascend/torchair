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
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F16(1024, 1024), F16(1024, 1024), "group", reduce_op="sum", bias=F16(1024), comm_turn=0),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_mm_all_reduce_base.default)
def convert_npu_mm_all_reduce_base(
    self: Tensor,
    x2: Tensor,
    hcom: str,
    *,
    reduce_op: str = 'sum',
    bias: Optional[Tensor] = None,
    comm_turn: int = 0,
    meta_outputs: TensorSpec = None
):
    # transpose_x1 is set to False by default
    transpose_x1 = False
    transpose_x2 = False
    '''NB: npu::npu_mm_all_reduce_base(Tensor self, Tensor x2, str hcom, *, str reduce_op='sum',
       Tensor? bias=None, int comm_turn=0) -> Tensor'''
    return ge.MatmulAllReduce(self,
                              x2,
                              bias=bias,
                              group=hcom,
                              reduce_op=reduce_op,
                              is_trans_a=transpose_x1,
                              is_trans_b=transpose_x2,
                              comm_turn=comm_turn)