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


@register_fx_node_ge_converter(torch.ops.aten.rot90.default)
def conveter_aten_rot90_default(
        self: Tensor,
        k: int = 1,
        dims: List[int] = [0, 1],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::rot90(Tensor self, int k=1, int[] dims=[0, 1]) -> Tensor """
    raise NotImplementedError("torch.ops.aten.rot90.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.rot90.out)
def conveter_aten_rot90_out(
        self: Tensor,
        k: int = 1,
        dims: List[int] = [0, 1],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::rot90.out(Tensor self, int k=1, int[] dims=[0, 1], *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.rot90.out ge converter is not implement!")


