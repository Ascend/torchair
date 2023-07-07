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


@register_fx_node_ge_converter(torch.ops.aten._cdist_forward.default)
def conveter_aten__cdist_forward_default(
        x1: Tensor,
        x2: Tensor,
        p: float,
        compute_mode: Optional[int],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_cdist_forward(Tensor x1, Tensor x2, float p, int? compute_mode) -> Tensor """
    raise NotImplementedError("torch.ops.aten._cdist_forward.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._cdist_forward.out)
def conveter_aten__cdist_forward_out(
        x1: Tensor,
        x2: Tensor,
        p: float,
        compute_mode: Optional[int],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_cdist_forward.out(Tensor x1, Tensor x2, float p, int? compute_mode, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten._cdist_forward.out ge converter is not implement!")


