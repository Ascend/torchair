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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(8, 8, 8)),
    Support(F32(8, 1, 8, 1, 8, 1)),
    Support(F16(8, 1, 8, 1, 8, 1)),
    Support(I64(8, 1, 8, 1, 8, 1)),
    Support(I32(8, 1, 8, 1, 8, 1)),
    Support(I16(8, 1, 8, 1, 8, 1)),
    Support(BOOL(8, 1, 8, 1, 8, 1)),
])
@register_fx_node_ge_converter(torch.ops.aten.squeeze.default)
def conveter_aten_squeeze_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::squeeze(Tensor(a) self) -> Tensor(a)"""
    return ge.Squeeze(self)


@register_fx_node_ge_converter(torch.ops.aten.squeeze.dim)
def conveter_aten_squeeze_dim(self: Tensor, dim: int, meta_outputs: TensorSpec = None):
    """NB: aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)"""
    if self.symsize is not None and all([not isinstance(s, torch.SymInt) for s in self.symsize]):
        if self.symsize[dim] != 1:
            return self
        return ge.Squeeze(self, axis=[dim])
    return ge.SelectV2(ge.Equal(ge.Gather(ge.Shape(self), dim), ge.Cast(1, dst_type=DataType.DT_INT32)),
                       ge.Squeeze(self, axis=[dim]), self)


@register_fx_node_ge_converter(torch.ops.aten.squeeze.dims)
def conveter_aten_squeeze_dims(self: Tensor, dim: List[int], meta_outputs: TensorSpec = None):
    """NB: aten::squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.squeeze.dims ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.squeeze.dimname)
def conveter_aten_squeeze_dimname(self: Tensor, dim: str, meta_outputs: TensorSpec = None):
    """NB: aten::squeeze.dimname(Tensor(a) self, str dim) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.squeeze.dimname ge_converter is not implemented!")
