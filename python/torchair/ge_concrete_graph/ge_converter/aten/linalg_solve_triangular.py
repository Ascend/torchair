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


@register_fx_node_ge_converter(torch.ops.aten.linalg_solve_triangular.default)
def conveter_aten_linalg_solve_triangular_default(
        self: Tensor,
        B: Tensor,
        *,
        upper: bool,
        left: bool = True,
        unitriangular: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::linalg_solve_triangular(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False) -> Tensor """
    raise NotImplementedError("torch.ops.aten.linalg_solve_triangular.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_solve_triangular.out)
def conveter_aten_linalg_solve_triangular_out(
        self: Tensor,
        B: Tensor,
        *,
        upper: bool,
        left: bool = True,
        unitriangular: bool = False,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::linalg_solve_triangular.out(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.linalg_solve_triangular.out ge converter is not implement!")


