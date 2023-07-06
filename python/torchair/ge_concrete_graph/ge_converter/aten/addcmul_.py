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


@register_fx_node_ge_converter(torch.ops.aten.addcmul_.default)
def conveter_aten_addcmul__default(
        self: Tensor,
        tensor1: Tensor,
        tensor2: Tensor,
        *,
        value: Union[Number, Tensor] = 1,
        meta_outputs: Any = None):
    """ NB: aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.addcmul_.default ge converter is not implement!")


