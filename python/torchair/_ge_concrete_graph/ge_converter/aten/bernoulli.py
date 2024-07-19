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
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter, \
    register_checkpoint_func
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, get_ge_rng_state
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@register_checkpoint_func([torch.ops.aten.bernoulli.default, torch.ops.aten.bernoulli.p, 
                           torch.ops.aten.bernoulli.Tensor])
def bernoulli_checkpoint(
    self: Tensor,
    p: Optional[Union[float, Tensor]] = None,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None,
    rng_state: Optional[Tensor] = None
):
    shape = ge.Shape(self)
    if rng_state is None:
        seed, offset = get_ge_rng_state(philox_num=10, gen=generator)
    else:
        seed, offset = ge.Unpack(rng_state, num=2, axis=0)
    return (seed, offset), ge.StatelessBernoulli(shape, p, seed, offset, dtype=self.dtype)


@declare_supported(
    [
        Support(F32(4, 16, value_range=(0, 1))),
        Support(F16(4, 16, value_range=(0, 1))),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.bernoulli.default)
def conveter_aten_bernoulli_default(
    self: Tensor, *, generator: Optional[Generator] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor"""
    _, result = bernoulli_checkpoint(self, self, generator, meta_outputs, None)
    return result


@register_fx_node_ge_converter(torch.ops.aten.bernoulli.out)
def conveter_aten_bernoulli_out(
    self: Tensor,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.bernoulli.out is redundant before pytorch 2.1.0, "
                       "might be supported in future version.")


@declare_supported(
    [
        Support(F32(4, 16), 0),
        Support(F32(4, 16), 1),
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
    _, result = bernoulli_checkpoint(self, p, generator, meta_outputs, None)
    return result


@declare_supported(
    [
        Support(F32(4, 16), F32(4, 16, value_range=(0, 1))),
        Support(F16(4, 16), F16(4, 16, value_range=(0, 1))),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.bernoulli.Tensor)
def conveter_aten_bernoulli_Tensor(
    self: Tensor,
    p: Tensor,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bernoulli.Tensor(Tensor self, Tensor p, *, Generator? generator=None) -> Tensor"""
    _, result = bernoulli_checkpoint(self, p, generator, meta_outputs, None)
    return result


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
    raise RuntimeError("torch.ops.aten.bernoulli.Tensor_out is redundant before pytorch 2.1.0, "
                       "might be supported in future version.")


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
    raise RuntimeError("torch.ops.aten.bernoulli.float_out is redundant before pytorch 2.1.0, "
                       "might be supported in future version.")