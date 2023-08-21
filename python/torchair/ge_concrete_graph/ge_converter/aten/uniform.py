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


@register_fx_node_ge_converter(torch.ops.aten.uniform.default)
def conveter_aten_uniform_default(
    self: Tensor,
    from_: float = 0.0,
    to: float = 1.0,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::uniform(Tensor self, float from=0., float to=1., *, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.uniform.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.uniform.out)
def conveter_aten_uniform_out(
    self: Tensor,
    from_: float = 0.0,
    to: float = 1.0,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::uniform.out(Tensor self, float from=0., float to=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.uniform.out ge_converter is not implemented!")
