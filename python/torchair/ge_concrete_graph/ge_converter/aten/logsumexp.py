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
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F32(2, 4, 6, 6, 2), [1], True),
        Support(F32(2, 4, 6, 6, 2), [2], True),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.logsumexp.default)
def conveter_aten_logsumexp_default(
    self: Tensor, dim: List[int], keepdim: bool = False, meta_outputs: TensorSpec = None
):
    """NB: aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor"""
    maxes = ge.ReduceMax(self, dim, keep_dims=True)
    if keepdim:
        maxes_squeezed = maxes
    else:
        raise NotImplementedError("When keepdim is False, torch.ops.aten.logsumexp.default"
              "  ge_converter is not implemented!")
    maxes_squeezed = ge.MaskedFill(maxes_squeezed, ge.Equal(ge.Abs(maxes_squeezed), float("inf")), 0.)
    result = ge.ReduceLogSumExp(ge.Sub(self, maxes), dim, keep_dims=keepdim)
    result = ge.Add(result, maxes_squeezed)
    return result


@register_fx_node_ge_converter(torch.ops.aten.logsumexp.names)
def conveter_aten_logsumexp_names(
    self: Tensor, dim: List[str], keepdim: bool = False, meta_outputs: TensorSpec = None
):
    """NB: aten::logsumexp.names(Tensor self, str[1] dim, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.logsumexp.names ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.logsumexp.names_out)
def conveter_aten_logsumexp_names_out(
    self: Tensor,
    dim: List[str],
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::logsumexp.names_out(Tensor self, str[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logsumexp.names_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.logsumexp.out)
def conveter_aten_logsumexp_out(
    self: Tensor,
    dim: List[int],
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logsumexp.out ge_converter is not implemented!")
