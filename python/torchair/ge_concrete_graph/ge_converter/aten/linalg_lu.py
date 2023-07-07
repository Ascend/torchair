import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
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


@register_fx_node_ge_converter(torch.ops.aten.linalg_lu.default)
def conveter_aten_linalg_lu_default(
        A: Tensor,
        *,
        pivot: bool = True,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::linalg_lu(Tensor A, *, bool pivot=True) -> (Tensor P, Tensor L, Tensor U) """
    raise NotImplementedError("torch.ops.aten.linalg_lu.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_lu.out)
def conveter_aten_linalg_lu_out(
        A: Tensor,
        *,
        pivot: bool = True,
        P: Tensor = None,
        L: Tensor = None,
        U: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::linalg_lu.out(Tensor A, *, bool pivot=True, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) """
    raise NotImplementedError("torch.ops.aten.linalg_lu.out ge converter is not implement!")


