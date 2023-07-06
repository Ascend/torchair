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


@register_fx_node_ge_converter(torch.ops.aten.masked_fill_.Scalar)
def conveter_aten_masked_fill__Scalar(
        self: Tensor,
        mask: Tensor,
        value: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!) """
    return ge.MaskedFill(self, mask, value)


@register_fx_node_ge_converter(torch.ops.aten.masked_fill_.Tensor)
def conveter_aten_masked_fill__Tensor(
        self: Tensor,
        mask: Tensor,
        value: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.masked_fill_.Tensor ge converter is not implement!")


