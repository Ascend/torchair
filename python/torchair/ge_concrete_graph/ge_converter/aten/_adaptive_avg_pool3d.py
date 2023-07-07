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


@register_fx_node_ge_converter(torch.ops.aten._adaptive_avg_pool3d.default)
def conveter_aten__adaptive_avg_pool3d_default(
        self: Tensor,
        output_size: Union[List[int], Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_adaptive_avg_pool3d(Tensor self, SymInt[3] output_size) -> Tensor """
    raise NotImplementedError("torch.ops.aten._adaptive_avg_pool3d.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._adaptive_avg_pool3d.out)
def conveter_aten__adaptive_avg_pool3d_out(
        self: Tensor,
        output_size: Union[List[int], Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_adaptive_avg_pool3d.out(Tensor self, SymInt[3] output_size, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten._adaptive_avg_pool3d.out ge converter is not implement!")


