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


@register_fx_node_ge_converter(torch.ops.aten.rand.default)
def conveter_aten_rand_default(
    size: Union[List[int], Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.rand.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rand.generator)
def conveter_aten_rand_generator(
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::rand.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.rand.generator ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rand.names)
def conveter_aten_rand_names(
    size: Union[List[int], Tensor],
    *,
    names: Optional[List[str]],
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::rand.names(SymInt[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.rand.names ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rand.generator_with_names)
def conveter_aten_rand_generator_with_names(
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    names: Optional[List[str]],
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::rand.generator_with_names(SymInt[] size, *, Generator? generator, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.rand.generator_with_names ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rand.out)
def conveter_aten_rand_out(
    size: Union[List[int], Tensor], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::rand.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.rand.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rand.generator_out)
def conveter_aten_rand_generator_out(
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::rand.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.rand.generator_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rand.names_out)
def conveter_aten_rand_names_out(
    size: Union[List[int], Tensor],
    *,
    names: Optional[List[str]],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::rand.names_out(SymInt[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.rand.names_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rand.generator_with_names_out)
def conveter_aten_rand_generator_with_names_out(
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    names: Optional[List[str]],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::rand.generator_with_names_out(SymInt[] size, *, Generator? generator, str[]? names, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.rand.generator_with_names_out ge_converter is not implemented!")
