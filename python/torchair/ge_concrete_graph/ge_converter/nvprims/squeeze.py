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


@register_fx_node_ge_converter(torch.ops.nvprims.squeeze.default)
def conveter_nvprims_squeeze_default(
    a: Tensor, dimensions: List[int], meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: nvprims::squeeze(Tensor(a) a, int[] dimensions) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.nvprims.squeeze.default ge_converter is not implemented!")
