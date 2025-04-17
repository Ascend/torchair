from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.repeat.default)
def conveter_aten_repeat_default(
    self: Tensor, repeats: Union[List[int], Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::repeat(Tensor self, SymInt[] repeats) -> Tensor"""
    # TO DO: add check between self.rank and repeats length
    return ge.Tile(self, repeats)


@register_fx_node_ge_converter(torch.ops.aten.repeat.out)
def conveter_aten_repeat_out(
    self: Tensor,
    repeats: Union[List[int], Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::repeat.out(Tensor self, SymInt[] repeats, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.repeat.out ge_converter is not implemented!")
