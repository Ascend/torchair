from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
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
        Support(F16(1024, 1024), F16(1024, 1024), "group", reduce_op="sum", bias=F16(1024), comm_turn=0),
        Support(I8(1024, 1024), I8(1024, 1024), "group", reduce_op="sum", bias=I32(1024), dequant_scale=I64(1),
                comm_turn=0),
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
    antiquant_scale: Optional[Tensor] = None,
    antiquant_offset: Optional[Tensor] = None,
    x3: Optional[Tensor] = None,
    dequant_scale: Optional[Tensor] = None,
    pertoken_scale: Optional[Tensor] = None,
    comm_quant_scale_1: Optional[Tensor] = None,
    comm_quant_scale_2: Optional[Tensor] = None,
    antiquant_group_size: int = 0,
    comm_turn: int = 0,
    meta_outputs: TensorSpec = None
):
    # transpose_x1 is set to False by default
    transpose_x1 = False
    transpose_x2 = False
    '''NB: npu::npu_mm_all_reduce_base(Tensor x1, Tensor x2, str hcom, *, str reduce_op='sum', Tensor? bias=None,
                                       Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? x3=None,
                                       Tensor? dequant_scale=None, Tensor? pertoken_scale=None,
                                       Tensor? comm_quant_scale_1=None, Tensor? comm_quant_scale_2=None,
                                       int antiquant_group_size=0, int comm_turn=0) -> Tensor'''
    check_dtype(self, x2, bias=bias, x3=x3, antiquant_scale=antiquant_scale,
                antiquant_offset=antiquant_offset, dequant_scale=dequant_scale)
    return ge.MatmulAllReduce(self,
                              x2,
                              bias=bias,
                              x3=x3,
                              antiquant_scale=antiquant_scale,
                              antiquant_offset=antiquant_offset,
                              dequant_scale=dequant_scale,
                              pertoken_scale=pertoken_scale,
                              comm_quant_scale_1=comm_quant_scale_1,
                              comm_quant_scale_2=comm_quant_scale_2,
                              group=hcom,
                              reduce_op=reduce_op,
                              is_trans_a=transpose_x1,
                              is_trans_b=transpose_x2,
                              comm_turn=comm_turn,
                              antiquant_group_size=antiquant_group_size)


def check_dtype(x1: Tensor, x2: Tensor, bias: Optional[Tensor], x3: Optional[Tensor],
                antiquant_scale: Optional[Tensor], antiquant_offset: Optional[Tensor],
                dequant_scale: Optional[Tensor]):
    if (x1.dtype == DataType.DT_FLOAT16 or x1.dtype == DataType.DT_BF16) and \
        (x2.dtype == DataType.DT_FLOAT16 or x2.dtype == DataType.DT_BF16):
        if x2.dtype != x1.dtype:
            raise AssertionError(f"type of x1:{x1.dtype} and x2:{x2.dtype} must be same.")
        if bias is not None and bias.dtype != x1.dtype:
            raise AssertionError(f"type of x1:{x1.dtype} and bias:{bias.dtype} must be same.")
        if x3 is not None and x3.dtype != x1.dtype:
            raise AssertionError(f"type of x1:{x1.dtype} and x3:{x3.dtype} must be same.")