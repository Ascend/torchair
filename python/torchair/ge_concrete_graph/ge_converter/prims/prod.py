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


@register_fx_node_ge_converter(torch.ops.prims.prod.default)
def conveter_prims_prod_default(
        inp: Tensor,
        dims: Optional[List[int]],
        *,
        output_dtype: Optional[int] = None,
        meta_outputs: Any = None):
    """ NB: prims::prod(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor """
    raise NotImplementedError("torch.ops.prims.prod.default ge converter is not implement!")


