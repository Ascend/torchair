import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
)


@register_fx_node_ge_converter(torch.ops.aten.linalg_lu_solve.default)
def conveter_aten_linalg_lu_solve_default(
        LU: Tensor,
        pivots: Tensor,
        B: Tensor,
        *,
        left: bool = True,
        adjoint: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::linalg_lu_solve(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False) -> Tensor """
    raise NotImplementedError("torch.ops.aten.linalg_lu_solve.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_lu_solve.out)
def conveter_aten_linalg_lu_solve_out(
        LU: Tensor,
        pivots: Tensor,
        B: Tensor,
        *,
        left: bool = True,
        adjoint: bool = False,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::linalg_lu_solve.out(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.linalg_lu_solve.out ge converter is not implement!")


