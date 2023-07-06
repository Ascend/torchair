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


@register_fx_node_ge_converter(torch.ops.aten.randperm.default)
def conveter_aten_randperm_default(
        n: Union[int, Tensor],
        *,
        dtype: Optional[int] = 4,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        meta_outputs: Any = None):
    """ NB: aten::randperm(SymInt n, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.randperm.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.randperm.generator)
def conveter_aten_randperm_generator(
        n: Union[int, Tensor],
        *,
        generator: Optional[Generator],
        dtype: Optional[int] = 4,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        meta_outputs: Any = None):
    """ NB: aten::randperm.generator(SymInt n, *, Generator? generator, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.randperm.generator ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.randperm.out)
def conveter_aten_randperm_out(
        n: Union[int, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::randperm.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.randperm.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.randperm.generator_out)
def conveter_aten_randperm_generator_out(
        n: Union[int, Tensor],
        *,
        generator: Optional[Generator],
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::randperm.generator_out(SymInt n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.randperm.generator_out ge converter is not implement!")


