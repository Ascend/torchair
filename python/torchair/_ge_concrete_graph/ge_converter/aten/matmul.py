from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType


@register_fx_node_ge_converter(torch.ops.aten.matmul.default)
def conveter_aten_matmul_default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::matmul(Tensor self, Tensor other) -> Tensor"""
    if self.dtype == DataType.DT_INT8 or other.dtype == DataType.DT_INT8:
        raise RuntimeError("torch.ops.aten.matmul.default ge_converter is not support int8 dtype!")
    return ge.MatMul(self, other, None)


@register_fx_node_ge_converter(torch.ops.aten.matmul.out)
def conveter_aten_matmul_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.matmul.out ge_converter is not implemented!")
