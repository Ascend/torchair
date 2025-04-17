import functools
import operator
from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


def calc_shape_prod(size, dim):
    shape_prod = 1
    if len(size) == 0:
        shape_prod = 1
    elif len(dim) == 0:
        shape_prod = functools.reduce(operator.mul, size, 1)
    else:
        for d in dim:
            shape_prod *= size[d]
    return shape_prod


def get_output_size(tensor, ndim, dim, keepdim):
    shape = ge.Shape(tensor)
    shape = [ge.Gather(shape, i) for i in range(ndim)]
    if not isinstance(dim, list) or not all(isinstance(d, int) for d in dim):
        return None
    new_dim = [d % ndim for d in dim]
    new_dim.sort(reverse=True)
    for nd in new_dim:
        if keepdim:
            shape[nd] = 1
        else:
            del shape[nd]
    if not shape:
        return []
    shape = ge.Pack(shape, N=len(shape), axis=0)
    return shape


def process_special_shape(tensor, dim, correction, keepdim, dtype):
    size = tensor.symsize
    if size is None:
        return None
    shape_prod = calc_shape_prod(size, dim)
    if isinstance(shape_prod, torch.SymInt):
        return None
    output_size = get_output_size(tensor, len(size), dim, keepdim)
    if output_size is None:
        return None
    if shape_prod == 0 or (shape_prod == 1 and not isinstance(correction, Tensor) and shape_prod <= correction):
        std = ge.Fill(output_size, ge.Cast(float('nan'), dst_type=dtype))
        return std
    if not isinstance(correction, Tensor) and correction > 1 and shape_prod <= correction:
        std = ge.Fill(output_size, ge.Cast(float('inf'), dst_type=dtype))
        return std
    return None


@register_fx_node_ge_converter(torch.ops.aten.std.default)
def conveter_aten_std_default(
    self: Tensor, unbiased: bool = True, meta_outputs: TensorSpec = None
):
    """NB: aten::std(Tensor self, bool unbiased=True) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.std.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std.dim)
def conveter_aten_std_dim(
    self: Tensor,
    dim: Optional[List[int]],
    unbiased: bool = True,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::std.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.std.dim ge_converter is not implemented!")


@declare_supported(
    [
        Support(F32(2, 2), 0, correction=1, keepdim=False),
        Support(F32(16, 16), -1, correction=1, keepdim=True),
        Support(F32(4, 4, 4), [0, 2], correction=1, keepdim=False),
        Support(F32(4, 4, 4), [0, 1, 2], correction=1, keepdim=False),
        Support(F32(2, 2), 0, correction=0, keepdim=False),
        Support(F32(2, 0, 5), [1], keepdim=True),
        Support(F16(2, 0, 5), [1], keepdim=False),
        Support(F32(2, 1, 5), [1], keepdim=True, correction=2),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.std.correction)
def conveter_aten_std_correction(
    self: Tensor,
    dim: Optional[List[int]] = None,
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor"""
    correction = 1 if correction is None else correction
    std_or_none = process_special_shape(self, dim, correction, keepdim, meta_outputs.dtype)
    if std_or_none is not None:
        return std_or_none
    mean = ge.ReduceMean(self, axes=dim, keep_dims=keepdim)
    if len(dim) == 1 and dim[0] == -1:
        pass
    elif len(dim) != self.rank and keepdim is False:
        dim_copy = dim.copy()
        dim_copy.sort()
        for d in dim_copy:
            mean = ge.Unsqueeze(mean, axes=[d])
    mean_copy = ge.Expand(mean, ge.Shape(self))
    return ge.ReduceStdWithMean(self,
                                mean_copy,
                                dim=dim,
                                keepdim=keepdim,
                                correction=correction)


@register_fx_node_ge_converter(torch.ops.aten.std.names_dim)
def conveter_aten_std_names_dim(
    self: Tensor,
    dim: List[str],
    unbiased: bool = True,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::std.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.std.names_dim ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std.names_out)
def conveter_aten_std_names_out(
    self: Tensor,
    dim: List[str],
    unbiased: bool = True,
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::std.names_out(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.std.names_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std.out)
def conveter_aten_std_out(
    self: Tensor,
    dim: Optional[List[int]],
    unbiased: bool = True,
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::std.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.std.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std.correction_out)
def conveter_aten_std_correction_out(
    self: Tensor,
    dim: Optional[List[int]] = None,
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::std.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.std.correction_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std.correction_names)
def conveter_aten_std_correction_names(
    self: Tensor,
    dim: List[str],
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::std.correction_names(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.std.correction_names ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std.correction_names_out)
def conveter_aten_std_correction_names_out(
    self: Tensor,
    dim: List[str],
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::std.correction_names_out(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.std.correction_names_out ge_converter is not implemented!")
