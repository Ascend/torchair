from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, memory_format, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.grid_sampler_2d_backward.default)
def conveter_aten_grid_sampler_2d_backward_default(
    grad_output: Tensor,
    input: Tensor,
    grid: Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    output_mask: List[bool],
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.grid_sampler_2d_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.grid_sampler_2d_backward.out)
def conveter_aten_grid_sampler_2d_backward_out(
    grad_output: Tensor,
    input: Tensor,
    grid: Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    output_mask: List[bool],
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::grid_sampler_2d_backward.out(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))"""
    raise NotImplementedError("torch.ops.aten.grid_sampler_2d_backward.out ge_converter is not implemented!")
