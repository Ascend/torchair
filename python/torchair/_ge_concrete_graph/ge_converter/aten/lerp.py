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
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import F32, F16, BF16, Support


@declare_supported(
    [
        Support(F32(1024, 1024), F32(1024, 1024), 0.5),
        Support(F16(5,), F16(5,), 0.5),
        Support(BF16(5,), BF16(5,), 0.5),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.lerp.Scalar)
def conveter_aten_lerp_Scalar(
    self: Tensor, end: Tensor, weight: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor"""
    return ge.Lerp(self, end, weight)


@register_fx_node_ge_converter(torch.ops.aten.lerp.Tensor)
def conveter_aten_lerp_Tensor(
    self: Tensor, end: Tensor, weight: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.lerp.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.lerp.Scalar_out)
def conveter_aten_lerp_Scalar_out(
    self: Tensor,
    end: Tensor,
    weight: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.lerp.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.lerp.Tensor_out)
def conveter_aten_lerp_Tensor_out(
    self: Tensor,
    end: Tensor,
    weight: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.lerp.Tensor_out ge_converter is not implemented!")
