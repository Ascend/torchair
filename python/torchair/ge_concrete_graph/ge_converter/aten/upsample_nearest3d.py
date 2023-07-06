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


@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest3d.vec)
def conveter_aten_upsample_nearest3d_vec(
        input: Tensor,
        output_size: Optional[Union[List[int], Tensor]],
        scale_factors: Optional[List[float]],
        meta_outputs: Any = None):
    """ NB: aten::upsample_nearest3d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor """
    raise NotImplementedError("torch.ops.aten.upsample_nearest3d.vec ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest3d.default)
def conveter_aten_upsample_nearest3d_default(
        self: Tensor,
        output_size: Union[List[int], Tensor],
        scales_d: Optional[float] = None,
        scales_h: Optional[float] = None,
        scales_w: Optional[float] = None,
        meta_outputs: Any = None):
    """ NB: aten::upsample_nearest3d(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.upsample_nearest3d.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest3d.out)
def conveter_aten_upsample_nearest3d_out(
        self: Tensor,
        output_size: Union[List[int], Tensor],
        scales_d: Optional[float] = None,
        scales_h: Optional[float] = None,
        scales_w: Optional[float] = None,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::upsample_nearest3d.out(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.upsample_nearest3d.out ge converter is not implement!")


