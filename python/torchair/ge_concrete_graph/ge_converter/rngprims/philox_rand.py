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


@register_fx_node_ge_converter(torch.ops.rngprims.philox_rand.default)
def conveter_rngprims_philox_rand_default(
        size: Union[List[int], Tensor],
        seed: Tensor,
        offset: Tensor,
        stride: Optional[List[int]],
        device: Optional[Device] = None,
        dtype: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: rngprims::philox_rand(SymInt[] size, Tensor seed, Tensor offset, int[]? stride, Device? device=None, ScalarType? dtype=None) -> (Tensor, Tensor) """
    raise NotImplementedError("torch.ops.rngprims.philox_rand.default ge converter is not implement!")


