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
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest2d_backward.default)
def conveter_aten_upsample_nearest2d_backward_default(
    grad_output: Tensor,
    output_size: Union[List[int], Tensor],
    input_size: Union[List[int], Tensor],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::upsample_nearest2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.upsample_nearest2d_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest2d_backward.grad_input)
def conveter_aten_upsample_nearest2d_backward_grad_input(
    grad_output: Tensor,
    output_size: Union[List[int], Tensor],
    input_size: Union[List[int], Tensor],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::upsample_nearest2d_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.upsample_nearest2d_backward.grad_input ge_converter is not implemented!")
