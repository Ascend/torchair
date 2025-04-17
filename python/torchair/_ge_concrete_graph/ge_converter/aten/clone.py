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


@register_fx_node_ge_converter(torch.ops.aten.clone.default)
def conveter_aten_clone_default(
    self: Tensor, *, memory_format: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor"""
    if memory_format is not None and memory_format is not torch.contiguous_format:
        raise RuntimeError(
            "torch.ops.aten.clone.default have some unprocessed parameters or cases, "
            "memory_format = {}, torch.contiguous_format = {}".format(memory_format, torch.contiguous_format))

    return ge.Identity(self)


@register_fx_node_ge_converter(torch.ops.aten.clone.out)
def conveter_aten_clone_out(
    self: Tensor,
    *,
    memory_format: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::clone.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.clone.out ge_converter is "
                       "redundant before pytorch 2.1.0, might be supported in future version.")
