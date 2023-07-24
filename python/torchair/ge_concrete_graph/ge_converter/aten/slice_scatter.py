from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, memory_format, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.slice_scatter.default)
def conveter_aten_slice_scatter_default(
    self: Tensor,
    src: Tensor,
    dim: int = 0,
    start: Optional[Union[int, Tensor]] = None,
    end: Optional[Union[int, Tensor]] = None,
    step: Union[int, Tensor] = 1,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.slice_scatter.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.slice_scatter.out)
def conveter_aten_slice_scatter_out(
    self: Tensor,
    src: Tensor,
    dim: int = 0,
    start: Optional[Union[int, Tensor]] = None,
    end: Optional[Union[int, Tensor]] = None,
    step: Union[int, Tensor] = 1,
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::slice_scatter.out(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.slice_scatter.out ge_converter is not implemented!")
