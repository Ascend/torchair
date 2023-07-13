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


@register_fx_node_ge_converter(torch.ops.aten.narrow_copy.default)
def conveter_aten_narrow_copy_default(
        self: Tensor,
        dim: int,
        start: Union[int, Tensor],
        length: Union[int, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::narrow_copy(Tensor self, int dim, SymInt start, SymInt length) -> Tensor """
    raise NotImplementedError("torch.ops.aten.narrow_copy.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.narrow_copy.out)
def conveter_aten_narrow_copy_out(
        self: Tensor,
        dim: int,
        start: Union[int, Tensor],
        length: Union[int, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::narrow_copy.out(Tensor self, int dim, SymInt start, SymInt length, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.narrow_copy.out ge converter is not implement!")


