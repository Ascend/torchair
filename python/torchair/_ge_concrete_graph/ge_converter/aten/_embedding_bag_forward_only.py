from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType


def _get_mode_str(mode: int):
    if mode == 0:
        return "sum"
    elif mode == 2:
        return "max"
    return "mean"


@register_fx_node_ge_converter(torch.ops.aten._embedding_bag_forward_only.default)
def conveter_aten__embedding_bag_forward_only_default(
    weight: Tensor,
    indices: Tensor,
    offsets: Tensor,
    scale_grad_by_freq: bool = False,
    mode: int = 1,
    sparse: bool = False,
    per_sample_weights: Optional[Tensor] = None,
    include_last_offset: bool = False,
    padding_idx: int = -1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_embedding_bag_forward_only(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)"""
    mode_str = _get_mode_str(mode)
    indices = dtype_promote(indices, target_dtype=DataType.DT_INT32)
    offsets = dtype_promote(offsets, target_dtype=DataType.DT_INT32)
    return ge.EmbeddingBag(weight=weight, indices=indices, offsets=offsets, per_sample_weights=per_sample_weights,
                           mode=mode_str, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse,
                           include_last_offset=include_last_offset, padding_idx=padding_idx)


@register_fx_node_ge_converter(torch.ops.aten._embedding_bag_forward_only.out)
def conveter_aten__embedding_bag_forward_only_out(
    weight: Tensor,
    indices: Tensor,
    offsets: Tensor,
    scale_grad_by_freq: bool = False,
    mode: int = 0,
    sparse: bool = False,
    per_sample_weights: Optional[Tensor] = None,
    include_last_offset: bool = False,
    padding_idx: int = -1,
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    out2: Tensor = None,
    out3: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_embedding_bag_forward_only.out(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))"""
    raise AssertionError("torch.ops.aten._embedding_bag_forward_only.out is redundant before pytorch 2.1.0, "
                         "might be supported in furture version.")
