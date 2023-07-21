import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported
from torchair.ge_concrete_graph.supported_declaration import *
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.utils import dtype_promote
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


@declare_supported([
    Support([F32(2, 2), F32(2, 1)], dim=1),
    Support([F32(2, 2), F32(1, 2)], dim=0),
    Support([F32(2, 2), F16(1, 2)], dim=0),
])
@register_fx_node_ge_converter(torch.ops.aten.cat.default)
def conveter_aten_cat_default(
        tensors: List[Tensor],
        dim: int = 0,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cat(Tensor[] tensors, int dim=0) -> Tensor """
    tensors = [dtype_promote(arg, target_dtype = meta_outputs.dtype) for arg in tensors]
    return ge.ConcatD(tensors, concat_dim=dim, N=len(tensors))


@register_fx_node_ge_converter(torch.ops.aten.cat.names)
def conveter_aten_cat_names(
        tensors: List[Tensor],
        dim: str,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cat.names(Tensor[] tensors, str dim) -> Tensor """
    raise NotImplementedError("torch.ops.aten.cat.names ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cat.names_out)
def conveter_aten_cat_names_out(
        tensors: List[Tensor],
        dim: str,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cat.names_out(Tensor[] tensors, str dim, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.cat.names_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cat.out)
def conveter_aten_cat_out(
        tensors: List[Tensor],
        dim: int = 0,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.cat.out ge converter is not implement!")


