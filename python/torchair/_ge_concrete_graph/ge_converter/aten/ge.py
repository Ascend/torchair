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
from torchair._ge_concrete_graph.utils import dtype_promote


def check_support_dtype(self, other):
    self_dtype = self.dtype
    if isinstance(other, Tensor):
        other_dtype = other.dtype
    else:
        other_dtype = torch_type_to_ge_type(utils.type_to_dtype(type(other)))
    supported_dtype = [DataType.DT_BOOL, DataType.DT_UINT8, DataType.DT_INT8, DataType.DT_INT16, DataType.DT_INT32,
                       DataType.DT_INT64, DataType.DT_FLOAT16, DataType.DT_BF16, DataType.DT_FLOAT, DataType.DT_DOUBLE]
    if any(input_dtype not in supported_dtype for input_dtype in [self_dtype, other_dtype]):
        raise RuntimeError(f"torch.ops.aten.ge.Tensor ge_converter expect input dtype DT_BOOL, DT_UINT8, DT_INT8,"
                           f"DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE,"
                           f"but found unsupported input dtype: self dtype {self_dtype}, other dtype {other_dtype}")


@declare_supported([    
    Support(BOOL(2, 3), F64(2, 3)),
    Support(U8(2, 3), U8(2, 3)),
    Support(I8(2, 3), I8(2, 3)),
    Support(I16(2, 3), I16(2, 3)),
    Support(I32(2, 3), I32(2, 3)),
    Support(I64(2, 3), I64(2, 3)),
    Support(F16(2, 3), F16(2, 3)),
    Support(BF16(2, 3), BF16(2, 3)),
    Support(F32(2, 3), F32(2, 3)),
    Support(F64(2, 3), BF16(2, 3)),
    Support(F64(2, 1), BF16(2, 3)),
])
@register_fx_node_ge_converter(torch.ops.aten.ge.Tensor)
def conveter_aten_ge_Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::ge.Tensor(Tensor self, Tensor other) -> Tensor"""
    check_support_dtype(self, other)
    self_dtype = ge_type_to_torch_type(self.dtype)
    other_dtype = ge_type_to_torch_type(other.dtype)
    if self_dtype != other_dtype:
        calculate_dtype = utils.get_higher_dtype(self_dtype, other_dtype)
        self, other = dtype_promote(self, other, target_dtype=calculate_dtype)
    elif self.dtype == DataType.DT_BOOL:
        self, other = dtype_promote(self, other, target_dtype=DataType.DT_FLOAT)
    return ge.GreaterEqual(self, other)


@declare_supported([
    Support(F32(1024, 1024), 0),
    Support(F32(1024, 1024), 1.0),
])
@register_fx_node_ge_converter(torch.ops.aten.ge.Scalar)
def conveter_aten_ge_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::ge.Scalar(Tensor self, Scalar other) -> Tensor"""
    other = dtype_promote(other, target_dtype=self.dtype)
    return ge.GreaterEqual(self, other)


@register_fx_node_ge_converter(torch.ops.aten.ge.Scalar_out)
def conveter_aten_ge_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.ge.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.Tensor_out)
def conveter_aten_ge_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.ge.Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.int)
def conveter_aten_ge_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::ge.int(int a, int b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ge.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.float)
def conveter_aten_ge_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::ge.float(float a, float b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ge.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.int_float)
def conveter_aten_ge_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::ge.int_float(int a, float b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ge.int_float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.float_int)
def conveter_aten_ge_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::ge.float_int(float a, int b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ge.float_int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.default)
def conveter_aten_ge_default(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::ge(Scalar a, Scalar b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ge.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.str)
def conveter_aten_ge_str(a: str, b: str, meta_outputs: TensorSpec = None):
    """NB: aten::ge.str(str a, str b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ge.str ge_converter is not implemented!")
