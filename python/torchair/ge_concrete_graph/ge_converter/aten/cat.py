import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.fx2ge_converter import register_testcase
from torchair.ge_concrete_graph.testing_utils import *
from torchair.ge_concrete_graph.ge_graph import Tensor
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

@register_testcase([
    TestInput([F32(2, 2), F32(2, 1)], dim=1),
    TestInput([F32(2, 2), F32(1, 2)], dim=0),
])
@register_fx_node_ge_converter(torch.ops.aten.cat.default)
def conveter_aten_cat_default(
        tensors: List[Tensor],
        dim: int = 0,
        meta_outputs: Any = None):
    """ NB: aten::cat(Tensor[] tensors, int dim=0) -> Tensor """
    return ge.ConcatD(tensors, concat_dim=dim, N=len(tensors))


@register_fx_node_ge_converter(torch.ops.aten.cat.names)
def conveter_aten_cat_names(
        tensors: List[Tensor],
        dim: str,
        meta_outputs: Any = None):
    """ NB: aten::cat.names(Tensor[] tensors, str dim) -> Tensor """
    raise NotImplementedError("torch.ops.aten.cat.names ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cat.names_out)
def conveter_aten_cat_names_out(
        tensors: List[Tensor],
        dim: str,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::cat.names_out(Tensor[] tensors, str dim, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.cat.names_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cat.out)
def conveter_aten_cat_out(
        tensors: List[Tensor],
        dim: int = 0,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.cat.out ge converter is not implement!")


