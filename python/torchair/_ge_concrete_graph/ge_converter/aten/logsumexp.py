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
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


def get_squeeze_dims(dims, ndims):
    squeeze_dims = []
    if dims:
        dims = [dim + ndims if dim < 0 else dim for dim in dims]
        for dim in dims:
            if dim in squeeze_dims:
                raise RuntimeError(f"In logsumexp, dim {dim} appears multiple times in the list of dims.")
            squeeze_dims.append(dim)
    else:
        squeeze_dims.extend([i for i in range(ndims)])
    return squeeze_dims


def squeeze_multiple(self, dims, self_rank):
    dims_to_squeeze = get_squeeze_dims(dims, self_rank)
    self = ge.Squeeze(self, axis=dims_to_squeeze)
    return self


@declare_supported(
    [
        Support(F32(2, 4, 6, 6, 2), [1], True),
        Support(F16(2, 4, 6, 6, 2), [2], True),
        Support(U8(10, 8), [1], True),
        Support(I8(10, 8), [0], False),
        Support(I64(10, 8, 3), [0, 1, 2], False),
        Support(BOOL(10, 8, 3), [1, 2], False),
        Support(I8(10, 8, 3), [1, -1], False),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.logsumexp.default)
def conveter_aten_logsumexp_default(
    self: Tensor, dim: List[int], keepdim: bool = False, meta_outputs: TensorSpec = None
):
    """NB: aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor"""
    self_rank = self.rank
    self = dtype_promote(self, target_dtype=meta_outputs.dtype)
    dim_int64 = dtype_promote(dim, target_dtype=DataType.DT_INT64)
    maxes = ge.ReduceMax(self, dim_int64, keep_dims=True)
    if keepdim:
        maxes_squeezed = maxes
    else:
        maxes_squeezed = squeeze_multiple(maxes, dim, self_rank)
    mask = ge.Equal(ge.Abs(maxes_squeezed), ge.Const(float("inf"), dtype=meta_outputs.dtype))
    maxes_squeezed = ge.MaskedFill(maxes_squeezed, mask, 0.)
    result = ge.ReduceLogSumExp(ge.Sub(self, maxes), dim_int64, keep_dims=keepdim)
    result = ge.Add(result, maxes_squeezed)
    return result


@register_fx_node_ge_converter(torch.ops.aten.logsumexp.names)
def conveter_aten_logsumexp_names(
    self: Tensor, dim: List[str], keepdim: bool = False, meta_outputs: TensorSpec = None
):
    """NB: aten::logsumexp.names(Tensor self, str[1] dim, bool keepdim=False) -> Tensor"""
    raise RuntimeError("torch.ops.aten.logsumexp.names ge_converter is not supported!")


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
    raise RuntimeError("torch.ops.aten.logsumexp.names_out ge_converter is not supported!")


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
    raise RuntimeError("torch.ops.aten.logsumexp.out ge_converter is not supported!")
