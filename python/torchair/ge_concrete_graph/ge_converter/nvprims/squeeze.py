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


@register_fx_node_ge_converter(torch.ops.nvprims.squeeze.default)
def conveter_nvprims_squeeze_default(
        a: Tensor,
        dimensions: List[int],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: nvprims::squeeze(Tensor(a) a, int[] dimensions) -> Tensor(a) """
    raise NotImplementedError("torch.ops.nvprims.squeeze.default ge converter is not implement!")


