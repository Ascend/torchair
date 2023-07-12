import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided
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


@register_fx_node_ge_converter(torch.ops.aten.upsample_bilinear2d.vec)
def conveter_aten_upsample_bilinear2d_vec(
        input: Tensor,
        output_size: Optional[Union[List[int], Tensor]],
        align_corners: bool,
        scale_factors: Optional[List[float]],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor """
    raise NotImplementedError("torch.ops.aten.upsample_bilinear2d.vec ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.upsample_bilinear2d.default)
def conveter_aten_upsample_bilinear2d_default(
        self: Tensor,
        output_size: Union[List[int], Tensor],
        align_corners: bool,
        scales_h: Optional[float] = None,
        scales_w: Optional[float] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::upsample_bilinear2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.upsample_bilinear2d.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.upsample_bilinear2d.out)
def conveter_aten_upsample_bilinear2d_out(
        self: Tensor,
        output_size: Union[List[int], Tensor],
        align_corners: bool,
        scales_h: Optional[float] = None,
        scales_w: Optional[float] = None,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::upsample_bilinear2d.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.upsample_bilinear2d.out ge converter is not implement!")


