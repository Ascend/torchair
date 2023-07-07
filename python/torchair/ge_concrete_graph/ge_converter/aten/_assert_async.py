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


@register_fx_node_ge_converter(torch.ops.aten._assert_async.default)
def conveter_aten__assert_async_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_assert_async(Tensor self) -> () """
    raise NotImplementedError("torch.ops.aten._assert_async.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._assert_async.msg)
def conveter_aten__assert_async_msg(
        self: Tensor,
        assert_msg: str,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_assert_async.msg(Tensor self, str assert_msg) -> () """
    raise NotImplementedError("torch.ops.aten._assert_async.msg ge converter is not implement!")


