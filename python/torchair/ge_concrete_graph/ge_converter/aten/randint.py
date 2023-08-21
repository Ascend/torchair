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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.randint.default)
def conveter_aten_randint_default(
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    dtype: Optional[int] = 4,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint(SymInt high, SymInt[] size, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.randint.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randint.generator)
def conveter_aten_randint_generator(
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    dtype: Optional[int] = 4,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint.generator(SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.randint.generator ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randint.low)
def conveter_aten_randint_low(
    low: Union[int, Tensor],
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    dtype: Optional[int] = 4,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint.low(SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.randint.low ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randint.low_generator)
def conveter_aten_randint_low_generator(
    low: Union[int, Tensor],
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    dtype: Optional[int] = 4,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint.low_generator(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.randint.low_generator ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randint.out)
def conveter_aten_randint_out(
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint.out(SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.randint.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randint.generator_out)
def conveter_aten_randint_generator_out(
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint.generator_out(SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.randint.generator_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randint.low_out)
def conveter_aten_randint_low_out(
    low: Union[int, Tensor],
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint.low_out(SymInt low, SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.randint.low_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randint.low_generator_out)
def conveter_aten_randint_low_generator_out(
    low: Union[int, Tensor],
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint.low_generator_out(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.randint.low_generator_out ge_converter is not implemented!")
