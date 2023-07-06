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


@register_fx_node_ge_converter(torch.ops.prims.fft_c2r.default)
def conveter_prims_fft_c2r_default(
        self: Tensor,
        *,
        dim: List[int],
        last_dim_size: Union[int, Tensor],
        meta_outputs: Any = None):
    """ NB: prims::fft_c2r(Tensor self, *, int[] dim, SymInt last_dim_size) -> Tensor """
    raise NotImplementedError("torch.ops.prims.fft_c2r.default ge converter is not implement!")


