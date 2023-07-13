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


@register_fx_node_ge_converter(torch.ops.aten.topk.default)
def conveter_aten_topk_default(
        self: Tensor,
        k: Union[int, Tensor],
        dim: int = -1,
        largest: bool = True,
        sorted: bool = True,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices) """
    raise NotImplementedError("torch.ops.aten.topk.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.topk.values)
def conveter_aten_topk_values(
        self: Tensor,
        k: Union[int, Tensor],
        dim: int = -1,
        largest: bool = True,
        sorted: bool = True,
        *,
        values: Tensor = None,
        indices: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::topk.values(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices) """
    raise NotImplementedError("torch.ops.aten.topk.values ge converter is not implement!")


