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


@register_fx_node_ge_converter(torch.ops.aten.diag_embed.default)
def conveter_aten_diag_embed_default(
    self: Tensor,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.diag_embed.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.diag_embed.out)
def conveter_aten_diag_embed_out(
    self: Tensor,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::diag_embed.out(Tensor self, int offset=0, int dim1=-2, int dim2=-1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.diag_embed.out ge_converter is not implemented!")
