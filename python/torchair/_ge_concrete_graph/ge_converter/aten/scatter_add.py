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
    Support, T, BF16


@declare_supported(
    [
        Support(F32(3, 5), 0, T([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.int64), F32(2, 5)),
        Support(F32(10,), 0, T([2, 4, 8], dtype=torch.int64), F32(20,)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.scatter_add.default)
def conveter_aten_scatter_add_default(
    self: Tensor, dim: int, index: Tensor, src: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor"""
    return ge.ScatterElements(self, index, src, axis=dim, reduction='add')


@register_fx_node_ge_converter(torch.ops.aten.scatter_add.out)
def conveter_aten_scatter_add_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_add.out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.scatter_add.out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.scatter_add.dimname)
def conveter_aten_scatter_add_dimname(
    self: Tensor, dim: str, index: Tensor, src: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_add.dimname(Tensor self, str dim, Tensor index, Tensor src) -> Tensor"""
    raise RuntimeError("aten::scatter_add.dimname is not yet supported with named tensors. ")
