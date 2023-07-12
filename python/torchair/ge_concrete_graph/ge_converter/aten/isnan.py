import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided
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


@register_fx_node_ge_converter(torch.ops.aten.isnan.default)
def conveter_aten_isnan_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::isnan(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.isnan.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.isnan.out)
def conveter_aten_isnan_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::isnan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.isnan.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.isnan.float)
def conveter_aten_isnan_float(
        a: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::isnan.float(float a) -> bool """
    raise NotImplementedError("torch.ops.aten.isnan.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.isnan.complex)
def conveter_aten_isnan_complex(
        a: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::isnan.complex(complex a) -> bool """
    raise NotImplementedError("torch.ops.aten.isnan.complex ge converter is not implement!")


