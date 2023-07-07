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


@register_fx_node_ge_converter(torch.ops.aten.index_reduce_.default)
def conveter_aten_index_reduce__default(
        self: Tensor,
        dim: int,
        index: Tensor,
        source: Tensor,
        reduce: str,
        *,
        include_self: bool = True,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index_reduce_(Tensor(a!) self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.index_reduce_.default ge converter is not implement!")


