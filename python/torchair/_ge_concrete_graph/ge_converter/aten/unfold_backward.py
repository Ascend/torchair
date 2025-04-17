from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.unfold_backward.default)
def conveter_aten_unfold_backward_default(
    grad_in: Tensor,
    input_sizes: Union[List[int], Tensor],
    dim: int,
    size: int,
    step: int,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::unfold_backward(Tensor grad_in, SymInt[] input_sizes, int dim, int size, int step) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.unfold_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.unfold_backward.out)
def conveter_aten_unfold_backward_out(
    grad_in: Tensor,
    input_sizes: Union[List[int], Tensor],
    dim: int,
    size: int,
    step: int,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::unfold_backward.out(Tensor grad_in, SymInt[] input_sizes, int dim, int size, int step, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.unfold_backward.out ge_converter is not implemented!")
