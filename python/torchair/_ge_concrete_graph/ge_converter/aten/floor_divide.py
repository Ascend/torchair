from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote, specific_op_input_layout, specific_op_output_layout


@declare_supported([
    Support(F64(7, 16, 16), F64(7, 16, 16, value_range=(1, 10))),
    Support(F32(7, 16, 16), F32(16, 16, value_range=(1, 10))),
    Support(F16(7, 16, 16), F16(16, 16, value_range=(1, 10))),
    Support(I64(7, 16, 16), I64(16, 16, value_range=(1, 10))),
    Support(I32(7, 16, 16), I32(16, 16, value_range=(1, 10))),
    Support(I16(7, 16, 16), I16(16, 16, value_range=(1, 10))),
    Support(I8(7, 16, 16), I8(16, 16, value_range=(1, 10))),
    Support(U8(7, 16, 16), U8(16, 16, value_range=(1, 10))),
    Support(F64(7, 16, 16), I64(16, 16, value_range=(1, 10))),
    Support(I32(7, 16, 16), U8(16, 16, value_range=(1, 10))),
])
@register_fx_node_ge_converter(torch.ops.aten.floor_divide.default)
def conveter_aten_floor_divide_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::floor_divide(Tensor self, Tensor other) -> Tensor"""
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.FloorDiv(self, other)


@register_fx_node_ge_converter(torch.ops.aten.floor_divide.Scalar)
def conveter_aten_floor_divide_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.floor_divide.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.floor_divide.out)
def conveter_aten_floor_divide_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.floor_divide.out ge_converter is not implemented!")
