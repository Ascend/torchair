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


@register_fx_node_ge_converter(torch.ops.aten.grid_sampler_2d.default)
def conveter_aten_grid_sampler_2d_default(
        input: Tensor,
        grid: Tensor,
        interpolation_mode: int,
        padding_mode: int,
        align_corners: bool,
        meta_outputs: Any = None):
    """ NB: aten::grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor """
    raise NotImplementedError("torch.ops.aten.grid_sampler_2d.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.grid_sampler_2d.out)
def conveter_aten_grid_sampler_2d_out(
        input: Tensor,
        grid: Tensor,
        interpolation_mode: int,
        padding_mode: int,
        align_corners: bool,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::grid_sampler_2d.out(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.grid_sampler_2d.out ge converter is not implement!")


