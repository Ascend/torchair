from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
import torch._prims_common as utils
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, ge_type_to_torch_type, torch_type_to_ge_type
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, I8, \
    U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote, _display_ge_type


def check_support_dtype(self, other):
    self_dtype = self.dtype
    if isinstance(other, Tensor):
        other_dtype = other.dtype
    else:
        other_dtype = torch_type_to_ge_type(utils.type_to_dtype(type(other)))
    supported_dtype = [DataType.DT_BOOL, DataType.DT_UINT8, DataType.DT_INT8, DataType.DT_INT16, DataType.DT_INT32,
                       DataType.DT_INT64, DataType.DT_FLOAT16, DataType.DT_BF16, DataType.DT_FLOAT, DataType.DT_DOUBLE]
    if any(input_dtype not in supported_dtype for input_dtype in [self_dtype, other_dtype]):
        ge_dtype = _display_ge_type(other_dtype) if self_dtype in supported_dtype else _display_ge_type(self_dtype)
        raise RuntimeError(f"torch.ops.aten.le.Tensor ge_converter expect input dtype DT_BOOL, DT_UINT8, DT_INT8,"
                           f"DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE,"
                           f"but find input dtype {ge_dtype}")


@declare_supported([    
    Support(F32(2, 3), F32(2, 3)),
    Support(F16(2, 3), BF16(2, 3)),
])
@register_fx_node_ge_converter(torch.ops.aten.le.Tensor)
def conveter_aten_le_Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::le.Tensor(Tensor self, Tensor other) -> Tensor"""
    check_support_dtype(self, other)
    self_dtype = ge_type_to_torch_type(self.dtype)
    other_dtype = ge_type_to_torch_type(other.dtype)
    if self_dtype != other_dtype:
        calculate_dtype = utils.get_higher_dtype(self_dtype, other_dtype)
        self, other = dtype_promote(self, other, target_dtype=calculate_dtype)
    elif self.dtype == DataType.DT_BOOL:
        self, other = dtype_promote(self, other, target_dtype=DataType.DT_FLOAT)
    return ge.LessEqual(self, other)


@declare_supported([
    Support(F32(1024, 1024), 0),
    Support(U8(1024, 1024), 1224.58),
    Support(BF16(1024, 1024), 1.0),
])
@register_fx_node_ge_converter(torch.ops.aten.le.Scalar)
def conveter_aten_le_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::le.Scalar(Tensor self, Scalar other) -> Tensor"""
    check_support_dtype(self, other)
    self_dtype = ge_type_to_torch_type(self.dtype)
    if isinstance(other, Tensor):
        other_dtype = ge_type_to_torch_type(other.dtype)
    else:    
        other_dtype = utils.type_to_dtype(type(other))
    if self_dtype != other_dtype:
        calculate_dtype = utils.get_higher_dtype(self_dtype, other_dtype)
        self, other = dtype_promote(self, other, target_dtype=calculate_dtype)
    elif self.dtype == DataType.DT_BOOL:
        self, other = dtype_promote(self, other, target_dtype=DataType.DT_FLOAT)
    return ge.LessEqual(self, other)


@register_fx_node_ge_converter(torch.ops.aten.le.Scalar_out)
def conveter_aten_le_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.le.Scalar_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.le.Tensor_out)
def conveter_aten_le_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.le.Tensor_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.le.int)
def conveter_aten_le_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::le.int(int a, int b) -> bool"""
    raise RuntimeError("torch.ops.aten.le.int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.le.float)
def conveter_aten_le_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::le.float(float a, float b) -> bool"""
    raise RuntimeError("torch.ops.aten.le.float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.le.int_float)
def conveter_aten_le_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::le.int_float(int a, float b) -> bool"""
    raise RuntimeError("torch.ops.aten.le.int_float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.le.float_int)
def conveter_aten_le_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::le.float_int(float a, int b) -> bool"""
    raise RuntimeError("torch.ops.aten.le.float_int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.le.default)
def conveter_aten_le_default(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::le(Scalar a, Scalar b) -> bool"""
    raise RuntimeError("torch.ops.aten.le.default ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.le.str)
def conveter_aten_le_str(a: str, b: str, meta_outputs: TensorSpec = None):
    """NB: aten::le.str(str a, str b) -> bool"""
    raise RuntimeError("torch.ops.aten.le.str ge_converter is not supported!")
