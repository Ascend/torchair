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


@register_fx_node_ge_converter(torch.ops.prims.iota.default)
def conveter_prims_iota_default(
        length: Union[int, Tensor],
        *,
        start: Union[int, Tensor],
        step: Union[int, Tensor],
        dtype: int,
        device: Device,
        requires_grad: bool,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: prims::iota(SymInt length, *, SymInt start, SymInt step, ScalarType dtype, Device device, bool requires_grad) -> Tensor """
    raise NotImplementedError("torch.ops.prims.iota.default ge converter is not implement!")


