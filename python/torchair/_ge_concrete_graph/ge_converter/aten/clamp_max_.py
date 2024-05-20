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
from torchair._ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.clamp_max_.default)
def conveter_aten_clamp_max__default(
    self: Tensor, max: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp_max_.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_max_.Tensor)
def conveter_aten_clamp_max__Tensor(
    self: Tensor, max: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_max_.Tensor(Tensor(a!) self, Tensor max) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp_max_.Tensor ge_converter is not implemented!")