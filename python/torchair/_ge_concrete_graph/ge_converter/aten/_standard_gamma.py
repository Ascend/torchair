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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten._standard_gamma.default)
def conveter_aten__standard_gamma_default(
    self: Tensor, generator: Optional[Generator] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::_standard_gamma(Tensor self, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._standard_gamma.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._standard_gamma.out)
def conveter_aten__standard_gamma_out(
    self: Tensor,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_standard_gamma.out(Tensor self, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._standard_gamma.out ge_converter is not implemented!")
