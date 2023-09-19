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
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, DataType, get_ge_rng_state
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@register_fx_node_ge_converter(torch.ops.aten.bernoulli.default)
def conveter_aten_bernoulli_default(
    self: Tensor, *, generator: Optional[Generator] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.bernoulli.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bernoulli.out)
def conveter_aten_bernoulli_out(
    self: Tensor,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bernoulli.out ge_converter is not implemented!")


@declare_supported(
    [
        Support(F32(4, 16), 0.5),
        Support(F16(4, 16), 0.8),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.bernoulli.p)
def conveter_aten_bernoulli_p(
    self: Tensor,
    p: float,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor"""
    shape = ge.Shape(self)
    if generator is not None:
        raise NotImplementedError("torch.ops.aten.bernoulli currently not support assign generator")
    seed, offset = get_ge_rng_state(philox_num=10)
    return ge.StatelessBernoulli(shape, p, seed, offset, dtype=self.dtype)


@register_fx_node_ge_converter(torch.ops.aten.bernoulli.Tensor)
def conveter_aten_bernoulli_Tensor(
    self: Tensor,
    p: Tensor,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bernoulli.Tensor(Tensor self, Tensor p, *, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.bernoulli.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bernoulli.Tensor_out)
def conveter_aten_bernoulli_Tensor_out(
    self: Tensor,
    p: Tensor,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bernoulli.Tensor_out(Tensor self, Tensor p, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bernoulli.Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bernoulli.float_out)
def conveter_aten_bernoulli_float_out(
    self: Tensor,
    p: float = 0.5,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bernoulli.float_out(Tensor self, float p=0.5, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bernoulli.float_out ge_converter is not implemented!")
