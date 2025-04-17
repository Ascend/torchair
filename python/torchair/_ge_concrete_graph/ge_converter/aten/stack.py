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
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support([F32(5, 6), F32(5, 6)]),
        Support([F32(6, 3, 3), F32(6, 3, 3)], 3),
        Support([F32(2, 16, 2, 8), F32(2, 16, 2, 8), F32(2, 16, 2, 8)]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.stack.default)
def conveter_aten_stack_default(
    tensors: List[Tensor], dim: int = 0, meta_outputs: TensorSpec = None
):
    """NB: aten::stack(Tensor[] tensors, int dim=0) -> Tensor"""
    return ge.Pack(tensors, N=len(tensors), axis=dim)


@register_fx_node_ge_converter(torch.ops.aten.stack.out)
def conveter_aten_stack_out(
    tensors: List[Tensor], dim: int = 0, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.stack.out ge_converter is not implemented!")
