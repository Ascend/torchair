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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.conj_physical.default)
def conveter_aten_conj_physical_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::conj_physical(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.conj_physical.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.conj_physical.out)
def conveter_aten_conj_physical_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::conj_physical.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.conj_physical.out ge_converter is not implemented!")
