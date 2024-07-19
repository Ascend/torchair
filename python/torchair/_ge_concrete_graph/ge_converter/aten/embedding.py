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


@register_fx_node_ge_converter(torch.ops.aten.embedding.default)
def conveter_aten_embedding_default(
    weight: Tensor,
    indices: Tensor,
    padding_idx: Union[int, Tensor] = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor"""
    if scale_grad_by_freq:
        raise NotImplementedError(
            "torch.ops.aten.embedding.default input scale_grad_by_freq only supports False now, "
            "but input scale_grad_by_freq = True")

    if sparse:
        raise NotImplementedError(
            "torch.ops.aten.embedding.default input sparse only supports False now, "
            "but input sparse = True")
    return ge.GatherV2(weight, indices, [0])


@register_fx_node_ge_converter(torch.ops.aten.embedding.out)
def conveter_aten_embedding_out(
    weight: Tensor,
    indices: Tensor,
    padding_idx: Union[int, Tensor] = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::embedding.out(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.embedding.out ge_converter is not implemented!")
