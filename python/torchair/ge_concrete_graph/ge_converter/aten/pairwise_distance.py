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


@register_fx_node_ge_converter(torch.ops.aten.pairwise_distance.default)
def conveter_aten_pairwise_distance_default(
        x1: Tensor,
        x2: Tensor,
        p: float = 2.0,
        eps: float = 1e-06,
        keepdim: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pairwise_distance(Tensor x1, Tensor x2, float p=2., float eps=9.9999999999999995e-07, bool keepdim=False) -> Tensor """
    raise NotImplementedError("torch.ops.aten.pairwise_distance.default ge converter is not implement!")


