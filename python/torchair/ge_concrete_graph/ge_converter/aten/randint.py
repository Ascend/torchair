import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
)


@register_fx_node_ge_converter(torch.ops.aten.randint.default)
def conveter_aten_randint_default(
        high: Union[int, Tensor],
        size: Union[List[int], Tensor],
        *,
        dtype: Optional[int] = 4,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        meta_outputs: Any = None):
    """ NB: aten::randint(SymInt high, SymInt[] size, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.randint.default ge converter is not implement!")


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
        meta_outputs: Any = None):
    """ NB: aten::randint.generator(SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.randint.generator ge converter is not implement!")


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
        meta_outputs: Any = None):
    """ NB: aten::randint.low(SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.randint.low ge converter is not implement!")


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
        meta_outputs: Any = None):
    """ NB: aten::randint.low_generator(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.randint.low_generator ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.randint.out)
def conveter_aten_randint_out(
        high: Union[int, Tensor],
        size: Union[List[int], Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::randint.out(SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.randint.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.randint.generator_out)
def conveter_aten_randint_generator_out(
        high: Union[int, Tensor],
        size: Union[List[int], Tensor],
        *,
        generator: Optional[Generator],
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::randint.generator_out(SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.randint.generator_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.randint.low_out)
def conveter_aten_randint_low_out(
        low: Union[int, Tensor],
        high: Union[int, Tensor],
        size: Union[List[int], Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::randint.low_out(SymInt low, SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.randint.low_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.randint.low_generator_out)
def conveter_aten_randint_low_generator_out(
        low: Union[int, Tensor],
        high: Union[int, Tensor],
        size: Union[List[int], Tensor],
        *,
        generator: Optional[Generator],
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::randint.low_generator_out(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.randint.low_generator_out ge converter is not implement!")


