from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.utils import dtype_promote


def _hack_float32_into_complex64(self):
    real, imag = ge.SplitV(self, [1, -1], -1, num_split=2)
    real = ge.Squeeze(real, axis=[-1])
    imag = ge.Squeeze(imag, axis=[-1])
    real, imag = dtype_promote(real, imag, target_dtype=DataType.DT_FLOAT)
    complex_out = ge.Complex(real, imag)
    return complex_out


def _hack_complex64_into_float32(self):
    real = ge.Real(self)
    imag = ge.Imag(self)
    res = ge.Pack([real, imag], N=2, axis=-1)
    return res


def _fft_apply_normalization(self, normalization, rank, signal):
    output_size = ge.Shape(self)
    signal_numel = 1
    for s in range(signal):
        index = rank - 1 - s
        signal_numel = ge.Mul(ge.Gather(output_size, index), signal_numel)
    signal_numel = ge.Cast(signal_numel, dst_type=DataType.DT_COMPLEX64)
    if normalization == 1:
        signal_numel = ge.Sqrt(signal_numel)
        self = ge.RealDiv(self, signal_numel)
    elif normalization == 2:
        self = ge.RealDiv(self, signal_numel)
    return self


def _check_dim_valid(dim, rank):
    for d in dim:
        if d < 0 or d >= rank:
            return False
    return True


def _transpose_sort(dims, sorted_dim):
    for d_index in sorted_dim:
        dims[d_index] = sorted_dim.index(d_index)
    return dims


def _exe_fft_c2c_1d(self, dims, normalization, signal):
    self_float = _hack_complex64_into_float32(self)
    self_float = ge.Permute(self_float, order=dims)
    self_complex = _hack_float32_into_complex64(self_float)
    fft = ge.FFT(self_complex)
    fft = _fft_apply_normalization(fft, normalization, self.rank, signal)
    fft = _hack_complex64_into_float32(fft)
    res = _hack_float32_into_complex64(ge.Permute(fft, order=dims))
    return res


def _exe_fft_c2c_2d(self, dims, normalization, dim, signal):
    left, right = [], []
    for d in dims:
        if d in dim:
            right.append(d)
        else:
            left.append(d)
    sorted_dim = left + right
    self_float = _hack_complex64_into_float32(self)
    self_float = ge.Permute(self_float, order=sorted_dim)
    self_complex = _hack_float32_into_complex64(self_float)
    fft = ge.FFT2D(self_complex)
    fft = _fft_apply_normalization(fft, normalization, self.rank, signal)
    fft = _hack_complex64_into_float32(fft)
    res = _hack_float32_into_complex64(ge.Permute(fft, order=_transpose_sort(dims, sorted_dim)))
    return res


@register_fx_node_ge_converter(torch.ops.aten._fft_c2c.default)
def conveter_aten__fft_c2c_default(
    self: Tensor,
    dim: Union[List[int], Tensor],
    normalization: int,
    forward: bool,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_fft_c2c(Tensor self, SymInt[] dim, int normalization, bool forward) -> Tensor"""
    ndim = self.rank
    dims = [i for i in range(ndim)]
    if isinstance(dim, Tensor):
        raise NotImplementedError(
            "When dim is Tensor, torch.ops.aten._fft_c2c.default ge_converter is not implemented!")
    if not _check_dim_valid(dim, self.rank):
        raise AssertionError(f"torch.ops.aten._fft_c2c.default the dim {dim} is out of range of {ndim}.")
    if len(dim) == 1:
        tmp = dims[-1]
        dims[-1] = dim[0]
        dims[dim[0]] = tmp
        result = _exe_fft_c2c_1d(self, dims, normalization, 1)
    elif len(dim) == 2:
        result = _exe_fft_c2c_2d(self, dims, normalization, dim, 2)
    else:
        raise NotImplementedError("torch.ops.aten._fft_c2c.default ge_converter is not implemented, when len(dim)>2")
    return result


@register_fx_node_ge_converter(torch.ops.aten._fft_c2c.out)
def conveter_aten__fft_c2c_out(
    self: Tensor,
    dim: Union[List[int], Tensor],
    normalization: int,
    forward: bool,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_fft_c2c.out(Tensor self, SymInt[] dim, int normalization, bool forward, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten._fft_c2c.out ge_converter is not supported!")
