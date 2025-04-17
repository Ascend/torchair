from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.linalg_lu.default)
def conveter_aten_linalg_lu_default(
    A: Tensor, *, pivot: bool = True, meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_lu(Tensor A, *, bool pivot=True) -> (Tensor P, Tensor L, Tensor U)"""
    raise NotImplementedError("torch.ops.aten.linalg_lu.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_lu.out)
def conveter_aten_linalg_lu_out(
    A: Tensor,
    *,
    pivot: bool = True,
    P: Tensor = None,
    L: Tensor = None,
    U: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_lu.out(Tensor A, *, bool pivot=True, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)"""
    raise NotImplementedError("torch.ops.aten.linalg_lu.out ge_converter is not implemented!")
