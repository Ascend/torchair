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


@register_fx_node_ge_converter(torch.ops.aten.nansum.default)
def conveter_aten_nansum_default(
        self: Tensor,
        dim: Optional[List[int]] = None,
        keepdim: bool = False,
        *,
        dtype: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::nansum(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.nansum.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.nansum.out)
def conveter_aten_nansum_out(
        self: Tensor,
        dim: Optional[List[int]] = None,
        keepdim: bool = False,
        *,
        dtype: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::nansum.out(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.nansum.out ge converter is not implement!")


