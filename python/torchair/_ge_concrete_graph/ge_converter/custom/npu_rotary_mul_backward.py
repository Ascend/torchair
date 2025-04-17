from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, BF16, I64, I8, U8, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


def check_symsize(input1):
    return all(isinstance(item, int) for item in input1.symsize)


@declare_supported([
    Support(BF16(4, 2048, 40, 128), BF16(4, 2048, 40, 128), BF16(1, 2048, 1, 128), BF16(1, 2048, 1, 128)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_rotary_mul_backward.default)
def conveter_npu_rotary_mul_backward_default(
    grad: Tensor, 
    self: Tensor, 
    r1: Tensor, 
    r2: Tensor, 
    *, 
    need_backward: bool = True, 
    meta_outputs: List[TensorSpec] = None
):
    """NB: npu::npu_rotary_mul_backward(Tensor grad, Tensor self, Tensor r1, Tensor r2) -> (Tensor, Tensor, Tensor)"""
    if self.rank != 4 or r1.rank != 4 or r2.rank != 4:
        raise RuntimeError('The dim of input tensor should equal to four')
    check_list = [grad, self, r1, r2]
    if all(check_symsize(check_input) for check_input in check_list):
        check_support = True
        broadcast_dim_num = 1
        for i in range(self.rank):
            if self.symsize[i] != r1.symsize[i]:
                broadcast_dim_num = broadcast_dim_num * self.symsize[i]
            if broadcast_dim_num > 1024:
                check_support = False
                break
        if self.symsize[3] % 64 != 0 or not check_support:
            raise NotImplementedError("when the last dimension of x is not a multiple of 64, \
                torch.ops.npu.npu_rotary_mul_backward.default is not implemented")
        else:
            dx, dr1, dr2 = ge.RotaryMulGrad(self, r1, r2, grad, need_backward=need_backward)
        return dx, dr1, dr2
    else:
        raise NotImplementedError("when there is a dynamic shape, \
            torch.ops.npu.npu_rotary_mul_backward.default is not implemented")