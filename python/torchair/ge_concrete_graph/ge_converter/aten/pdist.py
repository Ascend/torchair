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


@register_fx_node_ge_converter(torch.ops.aten.pdist.default)
def conveter_aten_pdist_default(
        self: Tensor,
        p: float = 2.0,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pdist(Tensor self, float p=2.) -> Tensor """
    raise NotImplementedError("torch.ops.aten.pdist.default ge converter is not implement!")


