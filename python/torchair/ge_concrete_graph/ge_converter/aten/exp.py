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
from torchair.ge_concrete_graph.utils import dtype_promote

@declare_supported([
    Support(F32(2, 10))
])
@register_fx_node_ge_converter(torch.ops.aten.exp.default)
def conveter_aten_exp_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::exp(Tensor self) -> Tensor"""
    return ge.Exp(self)


@register_fx_node_ge_converter(torch.ops.aten.exp.out)
def conveter_aten_exp_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.exp.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.exp.int)
def conveter_aten_exp_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::exp.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.exp.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.exp.float)
def conveter_aten_exp_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::exp.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.exp.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.exp.complex)
def conveter_aten_exp_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::exp.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.exp.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.exp.Scalar)
def conveter_aten_exp_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::exp.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.exp.Scalar ge_converter is not implemented!")
