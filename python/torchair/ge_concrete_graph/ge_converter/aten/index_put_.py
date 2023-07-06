import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
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


@register_fx_node_ge_converter(torch.ops.aten.index_put_.default)
def conveter_aten_index_put__default(
        self: Tensor,
        indices: List[Optional[Tensor]],
        values: Tensor,
        accumulate: bool = False,
        meta_outputs: Any = None):
    """ NB: aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.index_put_.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index_put_.hacked_twin)
def conveter_aten_index_put__hacked_twin(
        self: Tensor,
        indices: List[Tensor],
        values: Tensor,
        accumulate: bool = False,
        meta_outputs: Any = None):
    """ NB: aten::index_put_.hacked_twin(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.index_put_.hacked_twin ge converter is not implement!")


